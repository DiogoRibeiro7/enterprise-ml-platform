# Enterprise ML Pipeline - Data Ingestion Service
# File: services/data_ingestion/service.py

import asyncio
import aioboto3
import aiofiles
import asyncpg
from typing import Dict, List, Any, Optional, AsyncIterator
from dataclasses import dataclass
from abc import ABC, abstractmethod
import structlog
from contextlib import asynccontextmanager
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from streaming_data_types import deserializer_pool
import confluent_kafka.admin
from confluent_kafka import Consumer, Producer
from redis import asyncio as aioredis
import hashlib
from datetime import datetime, timedelta

from core.pipeline_orchestrator import BasePipelineStage, ExecutionContext, StageResult, PipelineStage, ExecutionStatus

logger = structlog.get_logger()

@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    type: str  # 's3', 'postgres', 'kafka', 'api', 'streaming'
    connection_config: Dict[str, Any]
    schema_config: Optional[Dict[str, Any]] = None
    partitioning: Optional[Dict[str, Any]] = None
    quality_rules: Optional[List[Dict]] = None

@dataclass
class IngestionMetrics:
    """Metrics for data ingestion"""
    records_ingested: int = 0
    records_failed: int = 0
    bytes_processed: int = 0
    processing_time_ms: float = 0.0
    data_quality_score: float = 0.0
    throughput_records_per_sec: float = 0.0

class DataConnector(ABC):
    """Abstract base class for data connectors"""
    
    @abstractmethod
    async def connect(self) -> None:
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        pass
    
    @abstractmethod
    async def read_data(self, config: Dict[str, Any]) -> AsyncIterator[pd.DataFrame]:
        pass
    
    @abstractmethod
    async def get_schema(self) -> pa.Schema:
        pass

class S3DataConnector(DataConnector):
    """High-performance S3 data connector with parallel processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        self.s3_client = None
        self.logger = structlog.get_logger().bind(connector="s3")
    
    async def connect(self) -> None:
        self.session = aioboto3.Session()
        self.s3_client = await self.session.client('s3').__aenter__()
        
        # Verify connection
        try:
            await self.s3_client.head_bucket(Bucket=self.config['bucket'])
            self.logger.info("S3 connection established", bucket=self.config['bucket'])
        except Exception as e:
            self.logger.error("S3 connection failed", error=str(e))
            raise
    
    async def disconnect(self) -> None:
        if self.s3_client:
            await self.s3_client.__aexit__(None, None, None)
    
    async def read_data(self, config: Dict[str, Any]) -> AsyncIterator[pd.DataFrame]:
        """Read data with parallel processing and streaming"""
        
        # List objects with pagination
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        semaphore = asyncio.Semaphore(config.get('max_parallel_downloads', 10))
        
        async def download_and_process(obj_key: str) -> Optional[pd.DataFrame]:
            async with semaphore:
                try:
                    response = await self.s3_client.get_object(
                        Bucket=self.config['bucket'],
                        Key=obj_key
                    )
                    
                    # Stream data to avoid memory issues
                    data = await response['Body'].read()
                    
                    # Process based on file type
                    if obj_key.endswith('.parquet'):
                        return pd.read_parquet(io.BytesIO(data))
                    elif obj_key.endswith('.csv'):
                        return pd.read_csv(io.BytesIO(data))
                    elif obj_key.endswith('.json'):
                        return pd.read_json(io.BytesIO(data), lines=True)
                    
                except Exception as e:
                    self.logger.error("Failed to process object", key=obj_key, error=str(e))
                    return None
        
        # Process objects in parallel
        tasks = []
        async for page in paginator.paginate(
            Bucket=self.config['bucket'],
            Prefix=config.get('prefix', '')
        ):
            for obj in page.get('Contents', []):
                task = asyncio.create_task(download_and_process(obj['Key']))
                tasks.append(task)
                
                # Yield results in batches to control memory usage
                if len(tasks) >= config.get('batch_size', 50):
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, pd.DataFrame):
                            yield result
                    
                    tasks.clear()
        
        # Process remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, pd.DataFrame):
                    yield result
    
    async def get_schema(self) -> pa.Schema:
        """Infer schema from sample data"""
        # Implementation would sample data and infer schema
        pass

class PostgreSQLConnector(DataConnector):
    """High-performance PostgreSQL connector with connection pooling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool = None
        self.logger = structlog.get_logger().bind(connector="postgresql")
    
    async def connect(self) -> None:
        try:
            self.pool = await asyncpg.create_pool(
                **self.config,
                min_size=5,
                max_size=20,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            self.logger.info("PostgreSQL connection pool established")
        except Exception as e:
            self.logger.error("PostgreSQL connection failed", error=str(e))
            raise
    
    async def disconnect(self) -> None:
        if self.pool:
            await self.pool.close()
    
    async def read_data(self, config: Dict[str, Any]) -> AsyncIterator[pd.DataFrame]:
        """Read data with chunking for memory efficiency"""
        
        query = config.get('query')
        chunk_size = config.get('chunk_size', 10000)
        
        async with self.pool.acquire() as connection:
            # Get total count for progress tracking
            count_query = f"SELECT COUNT(*) FROM ({query}) as subq"
            total_records = await connection.fetchval(count_query)
            
            # Read data in chunks
            offset = 0
            while offset < total_records:
                chunk_query = f"{query} LIMIT {chunk_size} OFFSET {offset}"
                
                rows = await connection.fetch(chunk_query)
                
                if not rows:
                    break
                
                # Convert to DataFrame
                df = pd.DataFrame([dict(row) for row in rows])
                yield df
                
                offset += chunk_size
    
    async def get_schema(self) -> pa.Schema:
        """Get schema from database metadata"""
        # Implementation would query information_schema
        pass

class KafkaStreamConnector(DataConnector):
    """Real-time Kafka stream connector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consumer = None
        self.logger = structlog.get_logger().bind(connector="kafka")
    
    async def connect(self) -> None:
        consumer_config = {
            'bootstrap.servers': self.config['bootstrap_servers'],
            'group.id': self.config.get('group_id', 'ml-pipeline'),
            'auto.offset.reset': self.config.get('auto_offset_reset', 'latest'),
            'enable.auto.commit': False,
            'max.poll.interval.ms': 300000,
            'session.timeout.ms': 30000
        }
        
        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe(self.config['topics'])
        self.logger.info("Kafka consumer connected", topics=self.config['topics'])
    
    async def disconnect(self) -> None:
        if self.consumer:
            self.consumer.close()
    
    async def read_data(self, config: Dict[str, Any]) -> AsyncIterator[pd.DataFrame]:
        """Stream data from Kafka with micro-batching"""
        
        batch_size = config.get('batch_size', 1000)
        timeout = config.get('timeout_ms', 1000)
        
        batch = []
        
        while True:
            msg = self.consumer.poll(timeout=timeout/1000.0)
            
            if msg is None:
                if batch:
                    yield pd.DataFrame(batch)
                    batch.clear()
                continue
            
            if msg.error():
                self.logger.error("Kafka message error", error=msg.error())
                continue
            
            try:
                # Deserialize message
                data = deserializer_pool.deserialize(msg.value())
                batch.append(data)
                
                if len(batch) >= batch_size:
                    yield pd.DataFrame(batch)
                    batch.clear()
                    self.consumer.commit(msg)
                    
            except Exception as e:
                self.logger.error("Message processing error", error=str(e))
    
    async def get_schema(self) -> pa.Schema:
        """Get schema from Kafka schema registry"""
        # Implementation would query Confluent Schema Registry
        pass

class DataIngestionService:
    """Enterprise data ingestion service with caching and monitoring"""
    
    def __init__(
        self,
        cache_config: Optional[Dict[str, Any]] = None,
        quality_config: Optional[Dict[str, Any]] = None
    ):
        self.connectors = {}
        self.cache_client = None
        self.quality_config = quality_config or {}
        self.cache_config = cache_config or {}
        self.logger = structlog.get_logger().bind(service="data_ingestion")
        self.metrics = IngestionMetrics()
    
    async def initialize(self):
        """Initialize service components"""
        
        # Initialize Redis cache if configured
        if self.cache_config.get('enabled', False):
            self.cache_client = await aioredis.from_url(
                self.cache_config['redis_url'],
                decode_responses=True,
                max_connections=20
            )
            self.logger.info("Cache client initialized")
    
    async def shutdown(self):
        """Shutdown service components"""
        
        # Disconnect all connectors
        for connector in self.connectors.values():
            await connector.disconnect()
        
        # Close cache connection
        if self.cache_client:
            await self.cache_client.close()
    
    def register_connector(self, name: str, connector: DataConnector):
        """Register a data connector"""
        self.connectors[name] = connector
    
    async def ingest_data(
        self,
        source_config: DataSource,
        processing_config: Dict[str, Any]
    ) -> AsyncIterator[pd.DataFrame]:
        """Main data ingestion method with caching and quality checks"""
        
        connector = self.connectors.get(source_config.type)
        if not connector:
            raise ValueError(f"Connector not found for type: {source_config.type}")
        
        start_time = datetime.now()
        
        # Check cache first
        cache_key = self._generate_cache_key(source_config, processing_config)
        cached_data = await self._get_from_cache(cache_key)
        
        if cached_data is not None:
            self.logger.info("Returning cached data", cache_key=cache_key)
            yield cached_data
            return
        
        # Connect to data source
        await connector.connect()
        
        try:
            # Process data in streaming fashion
            async for batch in connector.read_data(processing_config):
                
                # Apply data quality checks
                if self.quality_config.get('enabled', True):
                    batch = await self._apply_quality_checks(batch, source_config.quality_rules)
                
                # Update metrics
                self.metrics.records_ingested += len(batch)
                self.metrics.bytes_processed += batch.memory_usage(deep=True).sum()
                
                # Cache processed batch if configured
                if self.cache_config.get('enabled', False):
                    await self._cache_data(cache_key, batch)
                
                yield batch
        
        finally:
            await connector.disconnect()
            
            # Update final metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.processing_time_ms = processing_time
            
            if processing_time > 0:
                self.metrics.throughput_records_per_sec = (
                    self.metrics.records_ingested / (processing_time / 1000)
                )
    
    async def _apply_quality_checks(
        self,
        data: pd.DataFrame,
        quality_rules: Optional[List[Dict]]
    ) -> pd.DataFrame:
        """Apply data quality checks and transformations"""
        
        if not quality_rules:
            return data
        
        quality_score = 1.0
        
        for rule in quality_rules:
            rule_type = rule.get('type')
            
            if rule_type == 'completeness':
                # Check for missing values
                missing_threshold = rule.get('threshold', 0.1)
                missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
                
                if missing_ratio > missing_threshold:
                    quality_score *= (1 - missing_ratio)
                    
                    # Apply imputation if configured
                    if rule.get('impute', False):
                        numeric_cols = data.select_dtypes(include=['number']).columns
                        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
                        
                        categorical_cols = data.select_dtypes(include=['object']).columns
                        data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
            
            elif rule_type == 'uniqueness':
                # Check for duplicates
                duplicate_threshold = rule.get('threshold', 0.05)
                duplicate_ratio = data.duplicated().sum() / len(data)
                
                if duplicate_ratio > duplicate_threshold:
                    quality_score *= (1 - duplicate_ratio)
                    
                    # Remove duplicates if configured
                    if rule.get('remove_duplicates', True):
                        data = data.drop_duplicates()
            
            elif rule_type == 'validity':
                # Check data types and ranges
                column = rule.get('column')
                if column in data.columns:
                    if rule.get('data_type') == 'numeric':
                        invalid_mask = pd.to_numeric(data[column], errors='coerce').isna()
                        invalid_ratio = invalid_mask.sum() / len(data)
                        quality_score *= (1 - invalid_ratio)
                    
                    if 'range' in rule:
                        min_val, max_val = rule['range']
                        out_of_range = (data[column] < min_val) | (data[column] > max_val)
                        out_of_range_ratio = out_of_range.sum() / len(data)
                        quality_score *= (1 - out_of_range_ratio)
        
        self.metrics.data_quality_score = quality_score
        return data
    
    def _generate_cache_key(self, source_config: DataSource, processing_config: Dict[str, Any]) -> str:
        """Generate cache key for data source and configuration"""
        key_data = {
            'source_name': source_config.name,
            'source_type': source_config.type,
            'config_hash': hashlib.md5(str(processing_config).encode()).hexdigest()
        }
        return f"ingestion:{hashlib.md5(str(key_data).encode()).hexdigest()}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve data from cache"""
        if not self.cache_client:
            return None
        
        try:
            cached_data = await self.cache_client.get(cache_key)
            if cached_data:
                # Deserialize DataFrame from cache
                return pd.read_json(cached_data, orient='records')
        except Exception as e:
            self.logger.warning("Cache retrieval failed", error=str(e))
        
        return None
    
    async def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache processed data"""
        if not self.cache_client:
            return
        
        try:
            # Serialize DataFrame for caching
            serialized_data = data.to_json(orient='records')
            
            # Set with expiration
            ttl = self.cache_config.get('ttl_seconds', 3600)
            await self.cache_client.setex(cache_key, ttl, serialized_data)
            
        except Exception as e:
            self.logger.warning("Cache storage failed", error=str(e))
    
    def get_metrics(self) -> IngestionMetrics:
        """Get current ingestion metrics"""
        return self.metrics
    
    def reset_metrics(self):
        """Reset metrics counters"""
        self.metrics = IngestionMetrics()

class DataIngestionStage(BasePipelineStage):
    """Pipeline stage for data ingestion"""
    
    def __init__(
        self,
        service: DataIngestionService,
        data_sources: List[DataSource]
    ):
        super().__init__("data_ingestion", PipelineStage.DATA_INGESTION)
        self.service = service
        self.data_sources = data_sources
    
    async def _execute_stage(self, context: ExecutionContext) -> StageResult:
        """Execute data ingestion stage"""
        
        try:
            # Initialize service
            await self.service.initialize()
            
            # Register connectors based on data sources
            await self._register_connectors()
            
            # Process all data sources
            ingested_datasets = {}
            
            for data_source in self.data_sources:
                self.logger.info("Processing data source", source=data_source.name)
                
                processing_config = context.config.get('data_ingestion', {}).get(data_source.name, {})
                
                # Collect all batches for this data source
                batches = []
                async for batch in self.service.ingest_data(data_source, processing_config):
                    batches.append(batch)
                
                # Combine batches if needed
                if batches:
                    combined_data = pd.concat(batches, ignore_index=True)
                    ingested_datasets[data_source.name] = combined_data
                    
                    # Store data artifacts
                    artifact_path = f"/tmp/ingested_data_{data_source.name}.parquet"
                    combined_data.to_parquet(artifact_path)
            
            # Get metrics
            metrics = self.service.get_metrics()
            
            return StageResult(
                stage=self.stage_type,
                status=ExecutionStatus.SUCCESS,
                output=ingested_datasets,
                artifacts={
                    f"data_{source.name}": f"/tmp/ingested_data_{source.name}.parquet"
                    for source in self.data_sources
                },
                metrics={
                    "records_ingested": metrics.records_ingested,
                    "records_failed": metrics.records_failed,
                    "bytes_processed": metrics.bytes_processed,
                    "data_quality_score": metrics.data_quality_score,
                    "throughput_rps": metrics.throughput_records_per_sec
                }
            )
            
        except Exception as e:
            self.logger.error("Data ingestion stage failed", error=str(e))
            raise e
    
    async def _register_connectors(self):
        """Register appropriate connectors based on data sources"""
        
        for data_source in self.data_sources:
            if data_source.type == 's3' and 's3' not in self.service.connectors:
                s3_connector = S3DataConnector(data_source.connection_config)
                self.service.register_connector('s3', s3_connector)
            
            elif data_source.type == 'postgres' and 'postgres' not in self.service.connectors:
                pg_connector = PostgreSQLConnector(data_source.connection_config)
                self.service.register_connector('postgres', pg_connector)
            
            elif data_source.type == 'kafka' and 'kafka' not in self.service.connectors:
                kafka_connector = KafkaStreamConnector(data_source.connection_config)
                self.service.register_connector('kafka', kafka_connector)
    
    async def validate(self, context: ExecutionContext) -> bool:
        """Validate data ingestion prerequisites"""
        
        # Check if all required data sources are configured
        for data_source in self.data_sources:
            source_config = context.config.get('data_ingestion', {}).get(data_source.name)
            if not source_config:
                self.logger.error("Missing configuration for data source", source=data_source.name)
                return False
        
        # Validate connection configurations
        for data_source in self.data_sources:
            if not data_source.connection_config:
                self.logger.error("Missing connection config for data source", source=data_source.name)
                return False
        
        return True
    
    async def cleanup(self, context: ExecutionContext) -> None:
        """Cleanup data ingestion resources"""
        await self.service.shutdown()

# Factory for creating data ingestion components
class DataIngestionFactory:
    """Factory for creating data ingestion components"""
    
    @staticmethod
    def create_service(config: Dict[str, Any]) -> DataIngestionService:
        """Create data ingestion service with configuration"""
        
        cache_config = config.get('cache', {})
        quality_config = config.get('quality', {})
        
        return DataIngestionService(
            cache_config=cache_config,
            quality_config=quality_config
        )
    
    @staticmethod
    def create_data_sources(config: Dict[str, Any]) -> List[DataSource]:
        """Create data sources from configuration"""
        
        data_sources = []
        
        for source_config in config.get('data_sources', []):
            data_source = DataSource(
                name=source_config['name'],
                type=source_config['type'],
                connection_config=source_config['connection'],
                schema_config=source_config.get('schema'),
                partitioning=source_config.get('partitioning'),
                quality_rules=source_config.get('quality_rules')
            )
            data_sources.append(data_source)
        
        return data_sources
    
    @staticmethod
    def create_stage(config: Dict[str, Any]) -> DataIngestionStage:
        """Create complete data ingestion stage"""
        
        service = DataIngestionFactory.create_service(config)
        data_sources = DataIngestionFactory.create_data_sources(config)
        
        return DataIngestionStage(service, data_sources)

# Configuration example
SAMPLE_CONFIG = {
    "cache": {
        "enabled": True,
        "redis_url": "redis://localhost:6379",
        "ttl_seconds": 3600
    },
    "quality": {
        "enabled": True
    },
    "data_sources": [
        {
            "name": "transactions",
            "type": "s3",
            "connection": {
                "bucket": "ml-data-bucket",
                "region": "us-west-2"
            },
            "partitioning": {
                "column": "date",
                "strategy": "daily"
            },
            "quality_rules": [
                {
                    "type": "completeness",
                    "threshold": 0.05,
                    "impute": True
                },
                {
                    "type": "uniqueness",
                    "threshold": 0.02,
                    "remove_duplicates": True
                },
                {
                    "type": "validity",
                    "column": "amount",
                    "data_type": "numeric",
                    "range": [0, 1000000]
                }
            ]
        },
        {
            "name": "user_events",
            "type": "kafka",
            "connection": {
                "bootstrap_servers": "kafka1:9092,kafka2:9092",
                "topics": ["user-events"],
                "group_id": "ml-pipeline-consumer"
            },
            "quality_rules": [
                {
                    "type": "completeness",
                    "threshold": 0.1
                }
            ]
        },
        {
            "name": "features",
            "type": "postgres",
            "connection": {
                "host": "postgres-host",
                "port": 5432,
                "database": "features_db",
                "user": "ml_user",
                "password": "secure_password"
            },
            "quality_rules": [
                {
                    "type": "completeness",
                    "threshold": 0.05
                }
            ]
        }
    ],
    "data_ingestion": {
        "transactions": {
            "prefix": "transactions/year=2024/",
            "batch_size": 100,
            "max_parallel_downloads": 5
        },
        "user_events": {
            "batch_size": 1000,
            "timeout_ms": 5000
        },
        "features": {
            "query": "SELECT * FROM feature_store WHERE created_at >= NOW() - INTERVAL '7 days'",
            "chunk_size": 10000
        }
    }
}
