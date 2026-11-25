-- Preserve historical test guards and admit explicitly declared inner validation.
CREATE OR REPLACE FUNCTION nirs.check_prediction() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE space text; expected smallint; annotation_status text; raw_label text;
        config jsonb; expected_part text := 'test';
BEGIN
    SELECT label_space,configuration INTO STRICT space,config FROM nirs.experiment_runs WHERE run_id=NEW.run_id;
    IF NEW.protocol='inner_validation' THEN
        IF config->>'role' NOT IN ('inner_validation_used_for_selection','inner_validation_rule_development')
           OR config->>'split_id' IS DISTINCT FROM NEW.split_id
           OR config->>'role' IS NULL
        THEN RAISE EXCEPTION 'Inner validation requires an explicitly bound development run'; END IF;
        expected_part := 'validation';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM nirs.split_members WHERE split_id=NEW.split_id
        AND dataset_version_id=NEW.dataset_version_id AND observation_id=NEW.observation_id
        AND fold=NEW.fold AND part=expected_part)
    THEN RAISE EXCEPTION 'Prediction does not belong to its declared evaluation part'; END IF;
    IF space='binary' THEN
        SELECT a.target_label,a.status INTO STRICT expected,annotation_status
        FROM nirs.dataset_members d JOIN nirs.annotations a USING(annotation_id)
        WHERE d.dataset_version_id=NEW.dataset_version_id AND d.observation_id=NEW.observation_id;
        IF NEW.status='ok' AND annotation_status<>'verified'
        THEN RAISE EXCEPTION 'A binary decision needs a verified pinned reference'; END IF;
        IF NEW.predicted_label IS NOT NULL AND NEW.predicted_label NOT IN (0,1)
        THEN RAISE EXCEPTION 'Invalid binary prediction'; END IF;
    ELSE
        SELECT c.source_label->>'label' INTO STRICT raw_label
        FROM nirs.dataset_members d JOIN nirs.occurrences c
          ON c.source_version_id=d.source_version_id AND c.record_id=d.record_id
        WHERE d.dataset_version_id=NEW.dataset_version_id AND d.observation_id=NEW.observation_id;
        IF raw_label !~ '^[0-4]$' THEN RAISE EXCEPTION 'Invalid legacy reference label'; END IF;
        expected := raw_label::smallint;
        IF NEW.predicted_label IS NOT NULL AND NEW.predicted_label NOT BETWEEN 0 AND 4
        THEN RAISE EXCEPTION 'Invalid legacy prediction'; END IF;
    END IF;
    IF NEW.true_label IS DISTINCT FROM expected
    THEN RAISE EXCEPTION 'Prediction target differs from pinned dataset reference'; END IF;
    RETURN NEW;
END;
$$;
