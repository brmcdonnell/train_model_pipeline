# Databricks notebook source
query = """
WITH immunization_counts_by_patient AS (
    SELECT
        Patient as PatientId,
        COUNT(*) ImmunizationCount
    FROM
        immunizations
    GROUP BY
        Patient
),
encounter_counts_by_patient AS (
    SELECT
        Patient as PatientId,
        COUNT(*) EncounterCount
    FROM
        encounters
    GROUP BY
        Patient
),
medication_counts_by_patient AS (
    SELECT
        Patient as PatientId,
        COUNT(*) MedicationCount
    FROM
        medications
    GROUP BY
        Patient
),
procedure_counts_by_patient AS (
    SELECT
        Patient as PatientId,
        COUNT(*) ProcedureCount
    FROM
        procedures
    GROUP BY
        Patient
),
claim_count_by_patient AS (
    SELECT
        PatientId,
        COUNT(*) ClaimCount
    FROM
        claims
    GROUP BY
        PatientId
)
SELECT
    -- Patient fields
    pa.Id as PatientId,
    pa.BirthDate,
    pa.DeathDate,
    pa.Marital,
    pa.Race,
    pa.Ethnicity,
    pa.Gender,
    pa.Zip,
    pa.Healthcare_Expenses,
    pa.Healthcare_Coverage,

    -- Count fields
    COALESCE(cc.ClaimCount, 0) as ClaimCount,
    COALESCE(i.ImmunizationCount, 0) as ImmunizationCount,
    COALESCE(en.EncounterCount, 0) as EncounterCount,
    COALESCE(med.MedicationCount, 0) as MedicationCount,
    COALESCE(proc.ProcedureCount, 0) as ProcedureCount
FROM
    patients pa
    LEFT JOIN claim_count_by_patient cc
        ON cc.PatientId = pa.Id
    LEFT JOIN immunization_counts_by_patient i
        ON i.PatientId = pa.Id
    LEFT JOIN encounter_counts_by_patient en
        ON en.PatientId = pa.Id
    LEFT JOIN medication_counts_by_patient med
        ON med.PatientId = pa.Id
    LEFT JOIN procedure_counts_by_patient proc
        ON proc.PatientId = pa.Id
"""

result = spark.sql(query)
result.write.mode("overwrite").saveAsTable("prepared_patient_data")
