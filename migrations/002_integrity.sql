-- SQL CHECK permits NULL; require a target explicitly for verified annotations.
ALTER TABLE nirs.annotations ADD CONSTRAINT annotations_target_required CHECK (
    (status='verified' AND target_label IS NOT NULL AND target_label IN (0,1)) OR
    (status IN ('needs_review','excluded') AND target_label IS NULL)
);
ALTER TABLE nirs.annotations ADD CONSTRAINT annotations_json_identity CHECK (
    data ?& ARRAY['annotation_id','revision','status','target_label'] AND
    data->>'revision'=revision::text
);
CREATE FUNCTION nirs.check_prediction() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE space text; expected smallint; annotation_status text; raw_label text;
BEGIN
    SELECT label_space INTO STRICT space FROM nirs.experiment_runs WHERE run_id=NEW.run_id;
    IF NOT EXISTS (SELECT 1 FROM nirs.split_members WHERE split_id=NEW.split_id
        AND dataset_version_id=NEW.dataset_version_id AND observation_id=NEW.observation_id
        AND fold=NEW.fold AND part='test')
    THEN RAISE EXCEPTION 'Prediction does not belong to this test split'; END IF;
    IF space='binary' THEN
        SELECT a.target_label,a.status INTO STRICT expected,annotation_status
        FROM nirs.dataset_members d JOIN nirs.annotations a USING(annotation_id)
        WHERE d.dataset_version_id=NEW.dataset_version_id AND d.observation_id=NEW.observation_id;
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
CREATE TRIGGER prediction_reference_check BEFORE INSERT OR UPDATE ON nirs.predictions
    FOR EACH ROW EXECUTE FUNCTION nirs.check_prediction();
