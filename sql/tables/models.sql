/* =========  REVISED SCHEMA: weights only in pounds (lb)  ========== */
/* All mass-related fields use **lb** to keep a single unit system.   */

CREATE SCHEMA IF NOT EXISTS hvac;

/* One row = one HVAC model specification */
CREATE TABLE hvac.spec_model (
  id                          BIGSERIAL PRIMARY KEY,          -- auto-increment

  /* Identification */
  brand                       TEXT      NOT NULL,
  model_code                  TEXT UNIQUE NOT NULL,
  catalog_number              TEXT,

  /* General information */
  refrigerant_type            TEXT,
  factory_charge_lb           NUMERIC(6,2),   -- factory charge in pounds

  /* Electrical specifications */
  voltage                     TEXT,           -- e.g. '208-230'
  phase                       SMALLINT,       -- 1 or 3
  frequency_hz                SMALLINT,
  rla_a                       NUMERIC(6,2),
  fla_a                       NUMERIC(6,2),

  /* Pressure ratings */
  high_side_psi               NUMERIC(6,1),
  low_side_psi                NUMERIC(6,1),
  test_pressure_psi           NUMERIC(6,1),

  /* Heating specifications */
  heating_input_btu           BIGINT,
  heating_output_btu          BIGINT,
  heating_efficiency_pct      NUMERIC(5,2),
  gas_type                    TEXT,           -- Natural / LP / None

  /* Gas supply pressures */
  gas_supply_min_inwc         NUMERIC(5,2),
  gas_supply_max_inwc         NUMERIC(5,2),
  manifold_pressure_inwc      NUMERIC(5,2),

  /* Gas flow (optional) */
  gas_input_min_btu           BIGINT,
  gas_input_max_btu           BIGINT,
  gas_output_cap_btu          BIGINT,
  gas_supply_inwc             NUMERIC(5,2),

  /* Airflow & temperature */
  air_temp_rise_f             NUMERIC(5,1),
  max_ext_static_inwc         NUMERIC(5,2),

  /* Cooling data */
  cooling_capacity_btu        BIGINT,
  ieer                        NUMERIC(5,2),
  eer                         NUMERIC(5,2),

  /* Compressor data */
  compressor_quantity         SMALLINT,
  compressor_type             TEXT,
  compressor_hz               SMALLINT,
  refrigerant_charge_lb       NUMERIC(6,2),   -- system charge in pounds

  /* Ambient limits */
  min_ambient_f               NUMERIC(5,1),
  max_ambient_f               NUMERIC(5,1),
  max_air_temp_f              NUMERIC(5,1),

  /* Installation & reference */
  installation_type           TEXT,
  datasheet_url               TEXT,
  notes                       TEXT
);
