"""
Couchbase Vector store index.

An index using Couchbase as a vector store.
"""
from datetime import timedelta
from typing import Any, List, Optional

import httpx

from llama_index.schema import BaseNode, MetadataMode, TextNode
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import node_to_metadata_dict

# Default batch size
DEFAULT_BATCH_SIZE = 100


class CouchbaseVectorStore(VectorStore):
    """
    Couchbase Vector Store.

    To use, you should have the ``couchbase`` python package installed.

    """

    stores_text: bool = True
    flat_metadata: bool = True

    def __init__(
        self,
        connection_string: str,
        db_username: str,
        db_password: str,
        bucket_name: str,
        scope_name: str,
        collection_name: str,
        index_name: str,
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        embedding_key: Optional[str] = None,
        metadata_key: Optional[str] = "metadata",
    ) -> None:
        """
        Initializes a connection to a Couchbase database.

        Args:
            connection_string (str): The connection string for the database.
            db_username (str): The username for the database.
            db_password (str): The password for the database.
            bucket_name (str): The name of the bucket.
            scope_name (str): The name of the scope.
            collection_name (str): The name of the collection.
            index_name (str): The name of the index.
            id_key (Optional[str], optional): The field for the document ID. Defaults to "id".
            text_key (Optional[str], optional): The field for the document text. Defaults to "text".
            embedding_key (Optional[str], optional): The field for the document embedding. Defaults to         "text_embedding".
            metadata_key (Optional[str], optional): The field for the document metadata. Defaults to "metadata".

        Returns:
            None
        """
        try:
            from couchbase.auth import PasswordAuthenticator
            from couchbase.cluster import Cluster
            from couchbase.options import ClusterOptions
        except ImportError as e:
            print(e)
            raise ImportError(
                "Could not import couchbase python package. "
                "Please install couchbase SDK  with `pip install couchbase`."
            )

        if not connection_string:
            raise ValueError("connection_string must be provided.")

        if not db_username:
            raise ValueError("db_username must be provided.")

        if not db_password:
            raise ValueError("db_password must be provided.")

        if not bucket_name:
            raise ValueError("bucket_name must be provided.")

        if not scope_name:
            raise ValueError("scope_name must be provided.")

        if not collection_name:
            raise ValueError("collection_name must be provided.")

        if not index_name:
            raise ValueError("index_name must be provided.")

        if not embedding_key:
            self._embedding_key = text_key + "_embedding"  # type: ignore

        self._id_key = id_key
        self._connection_string = connection_string
        self._db_username = db_username
        self._db_password = db_password
        self._bucket_name = bucket_name
        self._scope_name = scope_name
        self._collection_name = collection_name
        self._text_key = text_key
        self._index_name = index_name
        self._metadata_key = metadata_key

        auth = PasswordAuthenticator(
            self._db_username,
            self._db_password,
        )
        self._cluster: Cluster = Cluster(self._connection_string, ClusterOptions(auth))
        # Wait until the cluster is ready for use.
        self._cluster.wait_until_ready(timedelta(seconds=5))

        self._bucket = self._cluster.bucket(self._bucket_name)
        self._scope = self._bucket.scope(self._scope_name)
        self._collection = self._scope.collection(self._collection_name)

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        """
        Add nodes to the collection and return their document IDs.

        Args:
            nodes (List[BaseNode]): List of nodes to add.
            **kwargs (Any): Additional keyword arguments.
                batch_size (int): Size of the batch for batch insert.

        Returns:
            List[str]: List of document IDs for the added nodes.
        """
        from couchbase.exceptions import DocumentExistsException

        batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)
        documents_to_insert = []
        doc_ids = List[str]

        for node in nodes:
            metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )
            doc_id: str = node.node_id

            doc = {
                self._id_key: doc_id,
                self._text_key: node.get_content(metadata_mode=MetadataMode.NONE),
                self._embedding_key: node.embedding,
                self._metadata_key: metadata,
            }

            documents_to_insert.append({doc_id: doc})

        for i in range(0, len(documents_to_insert), batch_size):
            batch = documents_to_insert[i : i + batch_size]
            try:
                # convert the list of dicts to a single dict for batch insert
                insert_batch = {}
                for doc in batch:  # type: ignore
                    insert_batch.update(doc)

                # upsert the batch of documents into the collection
                result = self._collection.upsert_multi(insert_batch)
                if result.all_ok:
                    doc_ids.extend(batch[0].keys())  # type: ignore
            except DocumentExistsException as e:
                raise ValueError(f"Document already exists: {e}")
        return doc_ids  # type: ignore

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """
        Delete a document by its reference document ID.

        :param ref_doc_id: The reference document ID to be deleted.
        :param **kwargs: Additional keyword arguments.
        :type ref_doc_id: str
        :return: None
        :rtype: None
        """
        self._collection.remove(ref_doc_id)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Executes a query in the vector store and returns the result.

        Args:
            query (VectorStoreQuery): The query object containing the search parameters.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            VectorStoreQueryResult: The result of the query containing the top-k nodes, similarities, and ids.
        """
        db_host = self._connection_string.split("//")[-1].strip("/")

        search_query = {
            "fields": [self._text_key, self._metadata_key, self._id_key],
            "sort": ["-_score"],
            "limit": query.similarity_top_k,
            "query": {"match_none": {}},
            "knn": [
                {
                    "k": query.similarity_top_k * 10,
                    "field": self._embedding_key,
                    "vector": query.query_embedding,
                }
            ],
        }

        search_result = httpx.post(
            f"http://{db_host}:8094/api/bucket/{self._bucket_name}/scope/{self._scope_name}/index/{self._index_name}/query",
            json=search_query,
            auth=(self._db_username, self._db_password),
            headers={"Content-Type": "application/json"},
        )
        # print(f"Search result: {search_result}")

        top_k_nodes = []
        top_k_scores = []
        top_k_ids = []

        if search_result.status_code == 200:
            response_json = search_result.json()
            # print(f"Response JSON: {response_json}")
            results = response_json["hits"]
            # print(f"Response JSON: {results}")
            for result in results:
                text = result["fields"].pop(self._text_key)
                score = result["score"]
                metadata_dict = result["fields"]
                id = result["fields"].pop(self._id_key)

                node = TextNode(
                    text=text,
                    id=id,
                    score=score,
                    metadata_dict=metadata_dict,
                )

                top_k_nodes.append(node)
                top_k_scores.append(score)
                top_k_ids.append(id)
        else:
            raise ValueError(
                f"Request failed with status code {search_result.status_code}"
                " and error message: {search_result.text}"
            )

        return VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )

    @property
    def client(self) -> Any:
        """
        Property function to access the client attribute.
        """
        return self._cluster
