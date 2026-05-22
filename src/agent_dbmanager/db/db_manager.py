from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from agent_dbmanager.db.models import Base, Supplier, SupplierLocation
from logger import logger
from agent_dbmanager.db.utils.normalization import *

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


def execute_search_query(location: str):
    """
    Generates and executes a raw SQL query in the database based on location.
    Returns an array of dictionaries (the extracted results).
    """
    session = SessionLocal()
    try:

        sql_query = """
            SELECT s.name, s.website, s.description, s.email, s.phone, s.vat_number,
                sl.id AS location_id, sl.country, sl.city, sl.address
            FROM supplier s
            LEFT JOIN supplier_locations sl ON s.id = sl.supplier_id
            WHERE 1=1
        """
        params = {}
            
        if location and location.strip():
            sql_query += " AND sl.city ILIKE :loc"
            params["loc"] = f"%{location.strip()}%"

        result = session.execute(text(sql_query), params)
        
        suppliers = [dict(row._mapping) for row in result]
        logger.info(f"[AGENT-DBMANAGER] Found {len(suppliers)} suppliers for location '{location}'")
        
        return suppliers
    except Exception as e:
        logger.error(f"[AGENT-DBMANAGER] Error executing SQL search query: {e}")
        raise e
    finally:
        session.close()
        

def save_supplier_to_db(data: dict):
    session = SessionLocal()

    try:
        normalized_name = normalize_name(data.get("name"))
        normalized_website = normalize_website(data.get("website"))
        normalized_email = normalize_emails(data.get("email", []))
        normalized_phone = normalize_phones(data.get("phone", []))
        vat_number = (data.get("vat_number", "").strip())

        supplier = None
        match_reason = None  # <-- track why we matched
        action_taken = None  # <-- final action log

        # 1. Search by VAT
        if vat_number:
            supplier = session.query(Supplier).filter(
                Supplier.vat_number == vat_number
            ).first()
            if supplier:
                match_reason = "VAT"

        # 2. Search by EMAIL
        if not supplier and normalized_email:
            supplier = session.query(Supplier).filter(
                Supplier.normalized_email.op("&&")(normalized_email)
            ).first()
            if supplier:
                match_reason = "EMAIL"

        # 3. Search by PHONE
        if not supplier and normalized_phone:
            supplier = session.query(Supplier).filter(
                Supplier.normalized_phone.op("&&")(normalized_phone)
            ).first()
            if supplier:
                match_reason = "PHONE"

        # 4. Search by WEBSITE
        if not supplier and normalized_website:
            supplier = session.query(Supplier).filter(
                Supplier.normalized_website == normalized_website
            ).first()
            if supplier:
                match_reason = "WEBSITE"

        # 5. Search by NAME
        if not supplier and normalized_name:
            supplier = session.query(Supplier).filter(
                Supplier.normalized_name == normalized_name
            ).first()
            if supplier:
                match_reason = "NAME"

        # DUPLICATE FOUND → update only
        if supplier:

            added_locations = 0

            existing_locations = {
                (
                    loc.country.lower(),
                    loc.city.lower(),
                    (loc.address or "").lower()
                )
                for loc in supplier.locations
            }

            for loc in data.get("locations", []):
                key = (
                    loc.get("country", "").lower(),
                    loc.get("city", "").lower(),
                    loc.get("address", "").lower()
                )

                if key not in existing_locations:
                    supplier.locations.append(
                        SupplierLocation(
                            country=loc.get("country", ""),
                            city=loc.get("city", ""),
                            address=loc.get("address")
                        )
                    )
                    added_locations += 1

            supplier.normalized_email = list(set((supplier.normalized_email or []) + normalized_email))
            supplier.email = list(set((supplier.email or []) + data.get("email", [])))

            supplier.normalized_phone = list(set((supplier.normalized_phone or []) + normalized_phone))
            supplier.phone = list(set((supplier.phone or []) + data.get("phone", [])))

            session.commit()

            action_taken = (
                f"UPDATED supplier_id={supplier.id} "
                f"(matched_by={match_reason}, added_locations={added_locations})"
            )

            logger.info(f"[AGENT-DBMANAGER] {action_taken}")

            return supplier.id

        # NEW SUPPLIER
        supplier = Supplier(
            name=data.get("name", ""),
            normalized_name=normalized_name,
            website=data.get("website"),
            normalized_website=normalized_website,
            description=data.get("description", ""),
            email=data.get("email", []),
            normalized_email=normalized_email,
            phone=data.get("phone", []),
            normalized_phone=normalized_phone,
            vat_number=vat_number
        )

        for loc in data.get("locations", []):
            supplier.locations.append(
                SupplierLocation(
                    country=loc.get("country", ""),
                    city=loc.get("city", ""),
                    address=loc.get("address")
                )
            )

        session.add(supplier)
        session.commit()

        action_taken = f"CREATED supplier_id={supplier.id}"

        logger.info(f"[AGENT-DBMANAGER] {action_taken}")

        return supplier.id

    except Exception as e:
        session.rollback()
        logger.error(f"[AGENT-DBMANAGER] Error saving supplier: {e}")
        raise

    finally:
        session.close()