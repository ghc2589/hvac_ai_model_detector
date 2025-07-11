#pragma once
#include <string>
#include <optional>

struct HVACModel
{
    /* ---------- Identification ---------- */
    std::string brand;                       // e.g. "Samsung"
    std::string model_code;                  // e.g. "AM100CXVAFH"
    std::optional<std::string> catalog_number;

    /* ---------- Refrigerant data ---------- */
    std::string refrigerant_type;            // e.g. "R-410A"
    std::string factory_charge_lb;           // string keeps units if present
    std::string refrigerant_charge_lb;

    /* ---------- Electrical specs ---------- */
    std::string voltage;                     // e.g. "208-230V"
    int         phase          = 0;          // 1 or 3
    int         frequency_hz   = 0;          // 50 or 60
    std::string rla_a;                       // Running Load Amps
    std::string fla_a;                       // Full-Load Amps

    /* ---------- Pressure specs ---------- */
    std::string high_side_psi;
    std::string low_side_psi;
    std::string test_pressure_psi;

    /* ---------- Heating (optional) ---------- */
    std::optional<int>    heating_input_btu;
    std::optional<int>    heating_output_btu;
    std::optional<std::string> heating_efficiency_pct;
    std::optional<std::string> gas_type;
    std::optional<std::string> gas_supply_min_inwc;
    std::optional<std::string> gas_supply_max_inwc;
    std::optional<std::string> manifold_pressure_inwc;
    std::optional<int>    gas_input_min_btu;
    std::optional<int>    gas_input_max_btu;
    std::optional<int>    gas_output_cap_btu;
    std::optional<std::string> gas_supply_inwc;
    std::optional<std::string> air_temp_rise_f;
    std::optional<std::string> max_ext_static_inwc;

    /* ---------- Cooling ---------- */
    int cooling_capacity_btu = 0;
    std::optional<std::string> ieer;
    std::optional<std::string> eer;

    /* ---------- Compressor ---------- */
    int         compressor_quantity = 0;
    std::string compressor_type;
    std::optional<std::string> compressor_hz;

    /* ---------- Environmental ---------- */
    std::string min_ambient_f;
    std::string max_ambient_f;
    std::string max_air_temp_f;

    /* ---------- Installation / docs ---------- */
    std::string installation_type;
    std::string datasheet_url;
    std::string notes;
};
