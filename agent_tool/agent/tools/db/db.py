from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agent_tool.agent.tools.db.models import Base, Supplier, SupplierLocation
from agent_tool.logger import logger

DATABASE_URL = "postgresql+psycopg2://admin:admin@localhost:5432/suppliersearchagentdb"

engine = create_engine(
    DATABASE_URL,
    echo=False
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def init_db():
    Base.metadata.create_all(bind=engine)


def save_supplier_to_db(data: dict):

    session = SessionLocal()

    try:

        supplier = Supplier(
            name=data.get("name", ""),
            website=data.get("website", ""),
            description=data.get("description", ""),
            email=data.get("email", []),
            phone=data.get("phone", []),
            vat_number=data.get("vat_number", "")
        )

        session.add(supplier)
        session.flush()

        locations = data.get("locations", [])

        for loc in locations:

            location = SupplierLocation(
                supplier_id=supplier.id,
                country=loc.get("country", ""),
                city=loc.get("city", ""),
                address=loc.get("address", "")
            )

            session.add(location)

        session.commit()

        logger.info(f"[DB] Supplier saved successfully: {supplier.name}")

        return supplier.id

    except Exception as e:

        session.rollback()

        logger.error(f"[DB] Error saving supplier: {e}")

        raise e

    finally:
        session.close()