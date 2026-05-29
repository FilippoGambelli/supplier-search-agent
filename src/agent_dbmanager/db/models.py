from sqlalchemy import Column, Integer, Text, ForeignKey, TIMESTAMP, func
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Supplier(Base):
    __tablename__ = "supplier"

    id = Column(Integer, primary_key=True)

    name = Column(Text, nullable=False)
    normalized_name = Column(Text)

    website = Column(Text)
    normalized_website = Column(Text)

    description = Column(Text)

    category = Column(ARRAY(Text))

    embedding = Column(Vector(384))

    email = Column(ARRAY(Text))
    normalized_email = Column(ARRAY(Text))
    
    phone = Column(ARRAY(Text))
    normalized_phone = Column(ARRAY(Text))

    vat_number = Column(Text)

    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(
        TIMESTAMP,
        server_default=func.now(),
        onupdate=func.now()
    )

    locations = relationship(
        "SupplierLocation",
        back_populates="supplier",
        cascade="all, delete-orphan"
    )


class SupplierLocation(Base):
    __tablename__ = "supplier_locations"

    id = Column(Integer, primary_key=True)

    supplier_id = Column(
        Integer,
        ForeignKey("supplier.id", ondelete="CASCADE"),
        nullable=False
    )

    country = Column(Text)
    region = Column(Text)
    province = Column(Text)
    city = Column(Text)
    address = Column(Text)

    supplier = relationship(
        "Supplier",
        back_populates="locations"
    )