import logging
import time
import psycopg
from functools import wraps
from typing import Callable, TypeVar, Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import ConnectionPool, AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from app.config.settings import (
    postgresql_connection_string,
    DB_POOL_SIZE,
    DB_CONNECTION_RETRIES,
    DB_RETRY_DELAY,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_DB
)

logger = logging.getLogger(__name__)

# Type variable for function return types
T = TypeVar('T')

# Connection settings
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row
}

# Singleton instances
_connection_pool = None
_async_connection_pool = None
_postgres_saver = None
_async_postgres_saver = None
_postgres_store = None


def create_database_if_not_exists():
    """
    Crea la base de datos si no existe.
    Se conecta a la base de datos 'postgres' para crear la base de datos objetivo.
    """
    # Cadena de conexión a la base de datos 'postgres' (que siempre existe)
    admin_connection_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/postgres"
    
    try:
        # Conectar a la base de datos postgres
        with psycopg.connect(admin_connection_string, autocommit=True) as conn:
            with conn.cursor() as cur:
                # Verificar si la base de datos existe
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (POSTGRES_DB,)
                )
                
                if not cur.fetchone():
                    # La base de datos no existe, crearla
                    logger.info(f"Base de datos '{POSTGRES_DB}' no existe. Creándola...")
                    
                    # Crear la base de datos (no se puede usar parámetros aquí)
                    cur.execute(f'CREATE DATABASE "{POSTGRES_DB}"')
                    logger.info(f"Base de datos '{POSTGRES_DB}' creada exitosamente")
                else:
                    logger.info(f"Base de datos '{POSTGRES_DB}' ya existe")
                    
    except Exception as e:
        logger.error(f"Error al crear la base de datos: {str(e)}")
        raise


def with_retry(max_retries: int = DB_CONNECTION_RETRIES, delay: int = DB_RETRY_DELAY) -> Callable:
    """
    Decorator for retrying database operations with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay in seconds between retries

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    current_delay = delay * (2 ** (retries - 1))  # Exponential backoff

                    if retries < max_retries:
                        logger.warning(
                            f"Database operation failed: {str(e)}. "
                            f"Retrying in {current_delay}s ({retries}/{max_retries})..."
                        )
                        time.sleep(current_delay)
                    else:
                        logger.error(f"Failed after {max_retries} retry attempts: {str(e)}")
                        raise

            # This line should never be reached, but keeps type checkers happy
            raise RuntimeError("Unexpected control flow in retry decorator")

        return wrapper

    return decorator


def get_connection_pool() -> ConnectionPool:
    """
    Get or create a PostgreSQL connection pool.
    This is a singleton pattern to ensure we only have one connection pool.
    """
    global _connection_pool

    if _connection_pool is None:
        try:
            # Primero asegurar que la base de datos existe
            create_database_if_not_exists()
            
            # Create connection pool
            _connection_pool = ConnectionPool(
                conninfo=postgresql_connection_string,
                max_size=DB_POOL_SIZE,
                kwargs=connection_kwargs,
            )

            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            print(e)
            logger.error(f"Failed to initialize PostgreSQL connection pool: {str(e)}")
            raise

    return _connection_pool


async def get_async_connection_pool() -> AsyncConnectionPool:
    """
    Get or create an asynchronous PostgreSQL connection pool.
    This is a singleton pattern to ensure we only have one connection pool.
    """
    global _async_connection_pool

    if _async_connection_pool is None:
        try:
            # Primero asegurar que la base de datos existe
            create_database_if_not_exists()
            
            # Create async connection pool
            _async_connection_pool = AsyncConnectionPool(
                conninfo=postgresql_connection_string,
                max_size=DB_POOL_SIZE,
                kwargs=connection_kwargs,
            )

            logger.info("Async PostgreSQL connection pool initialized")
        except Exception as e:
            print(e)
            logger.error(f"Failed to initialize async PostgreSQL connection pool: {str(e)}")
            raise

    return _async_connection_pool


@with_retry()
def get_postgres_saver() -> PostgresSaver:
    """
    Get or create a PostgresSaver instance.
    This is a singleton pattern to ensure we only have one instance.
    """
    global _postgres_saver

    if _postgres_saver is None:
        try:
            pool = get_connection_pool()

            # Create PostgresSaver
            _postgres_saver = PostgresSaver(pool)

            # Initialize tables
            _postgres_saver.setup()
            logger.info("PostgreSQL saver initialized successfully")
        except Exception as e:
            print(e)
            logger.error(f"Failed to initialize PostgreSQL saver: {str(e)}")
            raise

    return _postgres_saver


async def get_async_postgres_saver() -> AsyncPostgresSaver:
    """
    Get or create an AsyncPostgresSaver instance.
    This is a singleton pattern to ensure we only have one instance.
    """
    global _async_postgres_saver

    if _async_postgres_saver is None:
        try:
            # Set the correct event loop policy for Windows
            import platform
            if platform.system() == "Windows":
                import asyncio
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

            pool = await get_async_connection_pool()

            # Create AsyncPostgresSaver
            _async_postgres_saver = AsyncPostgresSaver(pool)

            # Initialize tables
            await _async_postgres_saver.setup()
            logger.info("Async PostgreSQL saver initialized successfully")
        except Exception as e:
            print(e)
            logger.error(f"Failed to initialize async PostgreSQL saver: {str(e)}")
            raise

    return _async_postgres_saver


@with_retry()
def get_postgres_store() -> PostgresStore:
    """
    Get or create a PostgresStore instance.
    This is a singleton pattern to ensure we only have one instance.
    """
    global _postgres_store

    if _postgres_store is None:
        try:
            pool = get_connection_pool()

            # Create PostgresStore
            _postgres_store = PostgresStore(pool)

            # Initialize tables
            _postgres_store.setup()
            logger.info("PostgreSQL store initialized successfully")
        except Exception as e:
            print(e)
            logger.error(f"Failed to initialize PostgreSQL store: {str(e)}")
            raise

    return _postgres_store


def check_postgres_connection() -> bool:
    """
    Check if PostgreSQL connection is working.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        pool = get_connection_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                logger.info("PostgreSQL connection check successful")
                return result is not None
    except Exception as e:
        logger.error(f"PostgreSQL connection check failed: {str(e)}")
        return False


def close_postgres_connections() -> None:
    """
    Close all PostgreSQL connections and clean up resources.
    """
    global _connection_pool, _async_connection_pool, _postgres_saver, _async_postgres_saver, _postgres_store

    try:
        if _connection_pool:
            _connection_pool.close()
            _connection_pool = None
            logger.info("PostgreSQL connection pool closed")

        if _async_connection_pool:
            _async_connection_pool.close()
            _async_connection_pool = None
            logger.info("Async PostgreSQL connection pool closed")

        # Reset singletons
        _postgres_saver = None
        _async_postgres_saver = None
        _postgres_store = None

        logger.info("All PostgreSQL connections closed successfully")
    except Exception as e:
        logger.error(f"Error closing PostgreSQL connections: {str(e)}")
