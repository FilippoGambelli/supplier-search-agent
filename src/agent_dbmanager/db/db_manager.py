import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from agent_dbmanager.db.utils.embedding  import get_embedding
from agent_dbmanager.db.models import Base, Supplier, SupplierLocation
from logger import logger
from agent_dbmanager.db.utils.normalization import *

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://admin:admin@localhost:5432/suppliersearchagentdb"
)

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
        normalized_name = normalize_name(data.get("name"))
        normalized_website = normalize_website(data.get("website"))
        normalized_email = normalize_emails(data.get("email", []))
        normalized_phone = normalize_phones(data.get("phone", []))
        normalized_category = normalize_categories(data.get("category", []))
        vat_number = (data.get("vat_number", "").strip())

        embedding_text = build_supplier_embedding_text({
            "description": data.get("description", ""),
            "category": data.get("category", [])
        })
        embedding = get_embedding(embedding_text)

        supplier = None
        match_reason = None

        # DUPLICATE DETECTION
        # 1. VAT
        if vat_number:
            supplier = session.query(Supplier).filter(
                Supplier.vat_number == vat_number
            ).first()
            if supplier:
                match_reason = "VAT"

        # 2. EMAIL
        if not supplier and normalized_email:
            supplier = session.query(Supplier).filter(
                Supplier.normalized_email.op("&&")(normalized_email)
            ).first()
            if supplier:
                match_reason = "EMAIL"

        # 3. PHONE
        if not supplier and normalized_phone:
            supplier = session.query(Supplier).filter(
                Supplier.normalized_phone.op("&&")(normalized_phone)
            ).first()
            if supplier:
                match_reason = "PHONE"

        # 4. WEBSITE
        if not supplier and normalized_website:
            supplier = session.query(Supplier).filter(
                Supplier.normalized_website == normalized_website
            ).first()
            if supplier:
                match_reason = "WEBSITE"

        # 5. NAME
        if not supplier and normalized_name:
            supplier = session.query(Supplier).filter(
                Supplier.normalized_name == normalized_name
            ).first()
            if supplier:
                match_reason = "NAME"

        # DUPLICATE FOUND → UPDATE
        if supplier:

            added_locations = 0

            existing_locations = {
                (
                    loc.country.lower(),
                    (loc.region or "").lower(),
                    (loc.province or "").lower(),
                    loc.city.lower(),
                    (loc.address or "").lower()
                )
                for loc in supplier.locations
            }

            for loc in data.get("locations", []):

                key = (
                    loc.get("country", "").lower(),
                    loc.get("region", "").lower(),
                    loc.get("province", "").lower(),
                    loc.get("city", "").lower(),
                    loc.get("address", "").lower()
                )

                if key not in existing_locations:

                    supplier.locations.append(
                        SupplierLocation(
                            country=loc.get("country", ""),
                            region=loc.get("region"),
                            province=loc.get("province"),
                            city=loc.get("city", ""),
                            address=loc.get("address")
                        )
                    )

                    added_locations += 1

            supplier.normalized_email = list(set((supplier.normalized_email or []) + normalized_email))
            supplier.email = list(set((supplier.email or []) + data.get("email", [])))

            supplier.normalized_phone = list(set((supplier.normalized_phone or []) + normalized_phone))
            supplier.phone = list(set((supplier.phone or []) + data.get("phone", [])))

            supplier.category = list(set((supplier.category or []) + data.get("category", [])))
            supplier.normalized_category = list(set((supplier.normalized_category or []) + normalized_category))

            supplier.embedding = embedding

            session.commit()

            logger.info(
                f"[AGENT-DBMANAGER] UPDATED supplier_id={supplier.id} "
                f"(matched_by={match_reason}, added_locations={added_locations})"
            )

            return supplier.id

        # NEW SUPPLIER
        supplier = Supplier(
            name=data.get("name", ""),
            normalized_name=normalized_name,
            website=data.get("website"),
            normalized_website=normalized_website,
            description=data.get("description", ""),
            category=data.get("category", []),
            normalized_category=normalized_category,
            email=data.get("email", []),
            normalized_email=normalized_email,
            phone=data.get("phone", []),
            normalized_phone=normalized_phone,
            embedding=embedding,
            vat_number=vat_number
        )

        for loc in data.get("locations", []):

            supplier.locations.append(
                SupplierLocation(
                    country=loc.get("country", ""),
                    region=loc.get("region"),
                    province=loc.get("province"),
                    city=loc.get("city", ""),
                    address=loc.get("address")
                )
            )

        session.add(supplier)
        session.commit()

        logger.info(f"[AGENT-DBMANAGER] CREATED supplier_id={supplier.id}")

        return supplier.id

    except Exception as e:
        session.rollback()
        logger.error(f"[AGENT-DBMANAGER] Error saving supplier: {e}")
        raise

    finally:
        session.close()

def execute_search_query( country: str = None, region: str = None, province: str = None, city: str = None, semantic_query: str = None):
    """
    Search suppliers using hierarchical geographic filters:
    country → region → province → city
    + optional semantic ranking on supplier embedding
    """
    session = SessionLocal()

    try:
        query_embedding = None

        if semantic_query and semantic_query.strip():
            query_embedding = get_embedding(semantic_query)

        if query_embedding is not None:
            sql_query = """
                SELECT DISTINCT ON (s.id) s.name, s.website, s.description, s.email, s.phone, s.vat_number, s.category, s.embedding <=> CAST(:query_embedding AS vector) AS similarity,
                    sl.id AS location_id, sl.country, sl.region, sl.province, sl.city, sl.address
                FROM supplier s
                LEFT JOIN supplier_locations sl
                    ON s.id = sl.supplier_id
                WHERE 1=1
            """
        else:
            sql_query = """
                SELECT DISTINCT ON (s.id) s.name, s.website, s.description, s.email, s.phone, s.vat_number, s.category, NULL AS similarity,
                    sl.id AS location_id, sl.country, sl.region, sl.province, sl.city, sl.address
                FROM supplier s
                LEFT JOIN supplier_locations sl
                    ON s.id = sl.supplier_id
                WHERE 1=1
            """

        params = {}

        # COUNTRY filter
        if country and country.strip():
            sql_query += " AND sl.country ILIKE :country"
            params["country"] = f"%{country.strip()}%"

        # REGION filter
        if region and region.strip():
            sql_query += " AND sl.region ILIKE :region"
            params["region"] = f"%{region.strip()}%"

        # PROVINCE filter
        if province and province.strip():
            sql_query += " AND sl.province ILIKE :province"
            params["province"] = f"%{province.strip()}%"

        # CITY filter
        if city and city.strip():
            sql_query += " AND sl.city ILIKE :city"
            params["city"] = f"%{city.strip()}%"

        # EMBEDDING PARAM
        if query_embedding is not None:
            params["query_embedding"] = query_embedding
            sql_query += " ORDER BY s.id, s.embedding <=> CAST(:query_embedding AS vector)"
        else:
            # fallback ordering (no semantic search)
            sql_query += " ORDER BY s.id"

        result = session.execute(text(sql_query), params)

        suppliers = [dict(row._mapping) for row in result]

        logger.info(
            f"[AGENT-DBMANAGER] Found {len(suppliers)} suppliers "
            f"(country={country}, region={region}, province={province}, city={city}, semantic_query={semantic_query})"
        )

        return suppliers

    except Exception as e:
        logger.error(f"[AGENT-DBMANAGER] Error executing search query: {e}")
        raise
    finally:
        session.close()