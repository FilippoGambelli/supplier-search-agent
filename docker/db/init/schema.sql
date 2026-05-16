CREATE TABLE IF NOT EXISTS suppliers (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    
    name TEXT NOT NULL,
    website TEXT,
    
    description TEXT,
    
    email TEXT[],
    phone TEXT[],
    
    vat_number TEXT,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS supplier_locations (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    supplier_id INTEGER NOT NULL REFERENCES suppliers(id) ON DELETE CASCADE,

    country TEXT NOT NULL,
    city TEXT NOT NULL,
    address TEXT
);

/*INDEXS*/
CREATE INDEX idx_supplier_locations_country_city
ON supplier_locations (country, city);

CREATE INDEX idx_supplier_locations_city
ON supplier_locations (city);

CREATE INDEX idx_supplier_locations_supplier_id
ON supplier_locations (supplier_id);