---
meta:
  id: sanosat1
  endian: le
doc: |
  :field delimiter: sanosat1_telemetry.delimiter
  :field callsign: sanosat1_telemetry.callsign
  :field packet_type: sanosat1_telemetry.packet_type
  :field com_temperature: sanosat1_telemetry.com_temperature
  :field battery_voltage: sanosat1_telemetry.battery_voltage
  :field charging_current: sanosat1_telemetry.charging_current
  :field battery_temperature: sanosat1_telemetry.battery_temperature
  :field radiation_level: sanosat1_telemetry.radiation_level
  :field no_of_resets: sanosat1_telemetry.no_of_resets
  :field antenna_deployment_status: sanosat1_telemetry.antenna_deployment_status

seq:
  - id: sanosat1_telemetry
    type: sanosat1_telemetry_t
types:
  sanosat1_telemetry_t:
    seq:
      - id: delimiter
        type: s4
        valid:
          any-of:
            - 0x0000ffff
      - id: callsign
        type: strz
        size: 7
        encoding: ASCII
        valid:
          any-of:
            - '"AM9NPQ"'
      - id: packet_type
        type: s2
      - id: com_temperature
        type: s2
      - id: battery_voltage
        type: s2
        doc: '[mV]'
      - id: charging_current
        type: s2
        doc: '[mA]'
      - id: battery_temperature
        type: s2
      - id: radiation_level
        type: s2
        doc: '[uSv/hr]'
      - id: no_of_resets
        type: s2
      - id: antenna_deployment_status
        type: u1
