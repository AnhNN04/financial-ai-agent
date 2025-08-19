import json
import boto3
from typing import List, Optional, Dict, Any
from botocore.exceptions import ClientError
from .base import BaseEmbeddings
from ..base import HealthCheckable
from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings
logger = Logger.get_logger(__name__)


class CohereV3Embeddings(BaseEmbeddings, HealthCheckable):
    """AWS Bedrock Cohere v3 multilingual embeddings implementation"""

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        input_type: str = "search_document",
        embedding_type: str = "float"
    ):
        """
        Initialize Cohere v3 embeddings with AWS Bedrock
        
        Args:
            model_id: Bedrock model ID for Cohere v3 (optional, will use settings if not provided)
            region_name: AWS region (optional, will use settings if not provided)
            aws_access_key_id: AWS access key (optional, will use settings if not provided)
            aws_secret_access_key: AWS secret key (optional, will use settings if not provided)
            aws_session_token: AWS session token (optional)
            input_type: Type of input text (search_document, search_query, classification, clustering)
            embedding_type: Type of embedding output (float, int8, uint8, binary, ubinary)
        """
        # Use settings values as defaults if not provided
        self.model_id = model_id or settings.embeddings.cohere_model_id
        self.input_type = input_type or getattr(settings.embeddings, 'cohere_input_type', 'search_document')
        self.embedding_type = embedding_type or getattr(settings.embeddings, 'cohere_embedding_type', 'float')
        self.max_batch_size = getattr(settings.embeddings, 'cohere_max_batch_size', 96)
        
        # Get AWS credentials from settings if not provided
        region = region_name or settings.s3.aws_region
        access_key = aws_access_key_id or settings.s3.aws_access_key_id
        secret_key = aws_secret_access_key or settings.s3.aws_secret_access_key
        
        try:
            # Initialize Bedrock client
            session_kwargs = {"region_name": region}
            
            if access_key and secret_key:
                session_kwargs.update({
                    "aws_access_key_id": access_key,
                    "aws_secret_access_key": secret_key
                })
            
            self.bedrock_client = boto3.client("bedrock-runtime", **session_kwargs)
            logger.info(f"Initialized Bedrock client with model: {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise

    def _invoke_model(self, texts: List[str]) -> List[List[float]]:
        """Invoke Bedrock model to get embeddings"""
        try:
            # Prepare request body for Cohere v3
            body = {
                "texts": texts,
                "input_type": self.input_type,
                "embedding_types": [self.embedding_type]
            }
            
            # Invoke the model
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response["body"].read())
            
            # Extract embeddings based on embedding_type
            if self.embedding_type == "float":
                embeddings = response_body.get("embeddings", [])
            else:
                # For other types like int8, uint8, binary, ubinary
                embeddings = response_body.get(f"embeddings_{self.embedding_type}", [])
            
            return embeddings
            
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(f"Bedrock API error ({error_code}): {error_message}")
            raise
        except Exception as e:
            logger.error(f"Failed to invoke Bedrock model: {e}")
            raise

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        try:
            logger.debug(f"Generating embedding for single text of length: {len(text)}")
            embeddings = self._invoke_model([text])
            
            if not embeddings:
                raise ValueError("No embeddings returned from model")
                
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            if not texts:
                return []
                
            logger.debug(f"Generating embeddings for {len(texts)} documents")
            
            # Cohere v3 can handle batch requests efficiently
            # Use max_batch_size from settings
            max_batch_size = self.max_batch_size
            
            all_embeddings = []
            
            for i in range(0, len(texts), max_batch_size):
                batch_texts = texts[i:i + max_batch_size]
                batch_embeddings = self._invoke_model(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Processed batch {i//max_batch_size + 1}, texts: {len(batch_texts)}")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for documents: {e}")
            raise

    def set_input_type(self, input_type: str):
        """
        Change input type for different use cases
        
        Args:
            input_type: search_document, search_query, classification, clustering
        """
        valid_types = ["search_document", "search_query", "classification", "clustering"]
        if input_type not in valid_types:
            raise ValueError(f"Invalid input_type. Must be one of: {valid_types}")
            
        self.input_type = input_type
        logger.info(f"Input type changed to: {input_type}")

    def set_embedding_type(self, embedding_type: str):
        """
        Change embedding type
        
        Args:
            embedding_type: float, int8, uint8, binary, ubinary
        """
        valid_types = ["float", "int8", "uint8", "binary", "ubinary"]
        if embedding_type not in valid_types:
            raise ValueError(f"Invalid embedding_type. Must be one of: {valid_types}")
            
        self.embedding_type = embedding_type
        logger.info(f"Embedding type changed to: {embedding_type}")

    def get_model_info(self) -> dict:
        """Get information about the current model configuration"""
        return {
            "model_id": self.model_id,
            "input_type": self.input_type,
            "embedding_type": self.embedding_type,
            "max_batch_size": self.max_batch_size,
            "max_input_length": 512_000  # Cohere v3 supports up to 512k tokens
        }
    

    async def check_health(self) -> Dict[str, Any]:
        """Check the health of Cohere v3 embedding provider."""
        try:
            # Test with a simple embedding request
            test_text = ["Health check"]
            embeddings = self._invoke_model(test_text)
            embedding_length = len(embeddings[0]) if embeddings else 0
            return {
                "status": "healthy",
                "details": {
                    "message": "Cohere v3 Bedrock API connection successful",
                    "model_id": self.model_id,
                    "embedding_length": embedding_length
                }
            }
        except Exception as e:
            logger.error(f"Cohere v3 health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "details": {
                    "error": str(e),
                    "message": "Failed to connect to Cohere v3 Bedrock API"
                }
            }