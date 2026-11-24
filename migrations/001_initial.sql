CREATE TABLE nirs.sources (
    source_id text PRIMARY KEY,
    metadata jsonb NOT NULL
);
CREATE TABLE nirs.source_versions (
    source_version_id text PRIMARY KEY,
    source_id text NOT NULL REFERENCES nirs.sources,
    file_path text NOT NULL,
    sha256_bytes text NOT NULL CHECK (sha256_bytes ~ '^[0-9a-f]{64}$'),
    sha256_lf text NOT NULL CHECK (sha256_lf ~ '^[0-9a-f]{64}$'),
    columns_json jsonb NOT NULL,
    row_count integer NOT NULL CHECK (row_count >= 0),
    accepted_count integer NOT NULL CHECK (accepted_count >= 0),
    rejected_count integer NOT NULL CHECK (rejected_count >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (row_count = accepted_count + rejected_count),
    UNIQUE (source_id, sha256_bytes)
);
CREATE TABLE nirs.import_jobs (
    job_id text PRIMARY KEY,
    source_id text NOT NULL,
    source_version_id text REFERENCES nirs.source_versions,
    status text NOT NULL CHECK (status IN ('running','complete','failed')),
    details jsonb NOT NULL,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);
CREATE TABLE nirs.observations (
    observation_id text PRIMARY KEY,
    sql_text text NOT NULL,
    sql_sha256 text NOT NULL CHECK (sql_sha256 ~ '^[0-9a-f]{64}$'),
    unit_kind text NOT NULL CHECK (unit_kind IN ('full_sql','payload_fragment','non_sql','unknown')),
    context_json jsonb NOT NULL,
    grouping_json jsonb NOT NULL,
    execution_json jsonb NOT NULL,
    purpose text NOT NULL CHECK (purpose IN ('research','contract_fixture')),
    current_annotation_id text NOT NULL
);
CREATE INDEX observations_sql_hash ON nirs.observations(sql_sha256);
CREATE TABLE nirs.annotations (
    annotation_id text PRIMARY KEY,
    observation_id text NOT NULL REFERENCES nirs.observations,
    revision integer NOT NULL CHECK (revision > 0),
    supersedes text REFERENCES nirs.annotations,
    status text NOT NULL CHECK (status IN ('verified','needs_review','excluded')),
    target_label smallint,
    data jsonb NOT NULL,
    UNIQUE (observation_id, revision),
    UNIQUE (annotation_id, observation_id),
    CHECK ((status = 'verified' AND target_label IN (0,1)) OR
           (status IN ('needs_review','excluded') AND target_label IS NULL)),
    CHECK (data->>'annotation_id' = annotation_id),
    CHECK (data->>'status' = status),
    CHECK ((data->>'target_label') IS NOT DISTINCT FROM target_label::text)
);
ALTER TABLE nirs.observations ADD CONSTRAINT observation_current_annotation
    FOREIGN KEY (current_annotation_id, observation_id)
    REFERENCES nirs.annotations(annotation_id, observation_id) DEFERRABLE INITIALLY DEFERRED;
CREATE TABLE nirs.occurrences (
    source_version_id text NOT NULL REFERENCES nirs.source_versions,
    record_id text NOT NULL,
    observation_id text NOT NULL REFERENCES nirs.observations,
    source_name text NOT NULL,
    source_label jsonb NOT NULL,
    source_json jsonb NOT NULL,
    raw_record jsonb NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    PRIMARY KEY (source_version_id, record_id),
    UNIQUE (source_version_id, record_id, observation_id)
);
CREATE INDEX occurrences_observation ON nirs.occurrences(observation_id);
CREATE INDEX occurrences_source_name ON nirs.occurrences(source_name);
CREATE TABLE nirs.rejected_records (
    source_version_id text NOT NULL REFERENCES nirs.source_versions,
    record_id text NOT NULL,
    reason text NOT NULL,
    raw_record_json text NOT NULL,
    PRIMARY KEY (source_version_id, record_id)
);
CREATE TABLE nirs.dataset_versions (
    dataset_version_id text PRIMARY KEY,
    name text NOT NULL,
    status text NOT NULL CHECK (status IN ('building','frozen')),
    specification jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE nirs.dataset_members (
    dataset_version_id text NOT NULL REFERENCES nirs.dataset_versions,
    observation_id text NOT NULL REFERENCES nirs.observations,
    annotation_id text NOT NULL,
    source_version_id text NOT NULL,
    record_id text NOT NULL,
    ordinal integer NOT NULL,
    PRIMARY KEY (dataset_version_id, observation_id),
    UNIQUE (dataset_version_id, ordinal),
    FOREIGN KEY (annotation_id, observation_id) REFERENCES nirs.annotations(annotation_id, observation_id),
    FOREIGN KEY (source_version_id, record_id, observation_id)
      REFERENCES nirs.occurrences(source_version_id, record_id, observation_id)
);
CREATE TABLE nirs.sample_groups (
    group_id text PRIMARY KEY,
    kind text NOT NULL,
    metadata jsonb NOT NULL
);
CREATE TABLE nirs.group_members (
    group_id text NOT NULL REFERENCES nirs.sample_groups,
    observation_id text NOT NULL REFERENCES nirs.observations,
    PRIMARY KEY (group_id, observation_id)
);
CREATE TABLE nirs.splits (
    split_id text PRIMARY KEY,
    dataset_version_id text NOT NULL REFERENCES nirs.dataset_versions,
    specification jsonb NOT NULL,
    UNIQUE (split_id, dataset_version_id)
);
CREATE TABLE nirs.split_members (
    split_id text NOT NULL,
    dataset_version_id text NOT NULL,
    observation_id text NOT NULL,
    fold integer NOT NULL CHECK (fold >= 0),
    part text NOT NULL CHECK (part IN ('train','validation','test')),
    PRIMARY KEY (split_id, fold, observation_id),
    FOREIGN KEY (split_id, dataset_version_id) REFERENCES nirs.splits(split_id, dataset_version_id),
    FOREIGN KEY (dataset_version_id, observation_id) REFERENCES nirs.dataset_members(dataset_version_id, observation_id)
);
CREATE TABLE nirs.artifacts (
    artifact_id text PRIMARY KEY CHECK (artifact_id ~ '^[0-9a-f]{64}$'),
    relative_path text NOT NULL UNIQUE,
    byte_size bigint NOT NULL CHECK (byte_size >= 0),
    status text NOT NULL CHECK (status IN ('pending','ready','missing')),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE nirs.source_artifacts (
    source_version_id text NOT NULL REFERENCES nirs.source_versions,
    artifact_id text NOT NULL REFERENCES nirs.artifacts,
    role text NOT NULL,
    PRIMARY KEY (source_version_id, role)
);
CREATE TABLE nirs.representations (
    representation_id text PRIMARY KEY,
    dataset_version_id text NOT NULL REFERENCES nirs.dataset_versions,
    configuration jsonb NOT NULL
);
CREATE TABLE nirs.representation_artifacts (
    representation_id text NOT NULL REFERENCES nirs.representations,
    artifact_id text NOT NULL REFERENCES nirs.artifacts,
    role text NOT NULL,
    metadata jsonb NOT NULL,
    PRIMARY KEY (representation_id, role)
);
CREATE TABLE nirs.experiment_runs (
    run_id text PRIMARY KEY,
    dataset_version_id text NOT NULL REFERENCES nirs.dataset_versions,
    method text NOT NULL,
    label_space text NOT NULL CHECK (label_space IN ('legacy_five_class','binary')),
    status text NOT NULL CHECK (status IN ('prepared','running','complete','failed')),
    configuration jsonb NOT NULL,
    UNIQUE (run_id, dataset_version_id)
);
CREATE TABLE nirs.predictions (
    run_id text NOT NULL,
    dataset_version_id text NOT NULL,
    split_id text NOT NULL,
    observation_id text NOT NULL,
    protocol text NOT NULL,
    fold integer NOT NULL,
    true_label smallint,
    predicted_label smallint,
    status text NOT NULL CHECK (status IN ('ok','api_error','parse_error','not_run')),
    metadata jsonb NOT NULL,
    PRIMARY KEY (run_id, protocol, fold, observation_id),
    FOREIGN KEY (run_id, dataset_version_id) REFERENCES nirs.experiment_runs(run_id, dataset_version_id),
    FOREIGN KEY (dataset_version_id, observation_id) REFERENCES nirs.dataset_members(dataset_version_id, observation_id),
    FOREIGN KEY (split_id, dataset_version_id) REFERENCES nirs.splits(split_id, dataset_version_id),
    CHECK ((status = 'ok' AND predicted_label IS NOT NULL) OR
           (status <> 'ok' AND predicted_label IS NULL))
);
CREATE INDEX predictions_dataset ON nirs.predictions(dataset_version_id);
CREATE TABLE nirs.metrics (
    run_id text NOT NULL REFERENCES nirs.experiment_runs,
    protocol text NOT NULL,
    fold integer NOT NULL,
    metric_name text NOT NULL,
    value jsonb NOT NULL,
    PRIMARY KEY (run_id, protocol, fold, metric_name)
);
CREATE TABLE nirs.run_artifacts (
    run_id text NOT NULL REFERENCES nirs.experiment_runs,
    artifact_id text NOT NULL REFERENCES nirs.artifacts,
    role text NOT NULL,
    metadata jsonb NOT NULL,
    PRIMARY KEY (run_id, role)
);
CREATE FUNCTION nirs.reject_change() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'Immutable research record'; END;
$$;
CREATE TRIGGER annotations_immutable BEFORE UPDATE OR DELETE ON nirs.annotations
    FOR EACH ROW EXECUTE FUNCTION nirs.reject_change();
CREATE FUNCTION nirs.protect_dataset_members() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE version_id text;
BEGIN
    IF TG_OP = 'DELETE' THEN version_id := OLD.dataset_version_id;
    ELSE version_id := NEW.dataset_version_id; END IF;
    IF EXISTS (SELECT 1 FROM nirs.dataset_versions WHERE dataset_version_id=version_id AND status='frozen')
       OR (TG_OP='UPDATE' AND EXISTS (SELECT 1 FROM nirs.dataset_versions WHERE dataset_version_id=OLD.dataset_version_id AND status='frozen'))
    THEN RAISE EXCEPTION 'Dataset is frozen'; END IF;
    IF TG_OP='DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
END;
$$;
CREATE TRIGGER dataset_members_immutable BEFORE INSERT OR UPDATE OR DELETE ON nirs.dataset_members
    FOR EACH ROW EXECUTE FUNCTION nirs.protect_dataset_members();
CREATE FUNCTION nirs.protect_dataset_version() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.status='frozen' THEN RAISE EXCEPTION 'Dataset is frozen'; END IF;
    IF TG_OP='DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
END;
$$;
CREATE TRIGGER frozen_dataset_immutable BEFORE UPDATE OR DELETE ON nirs.dataset_versions
    FOR EACH ROW EXECUTE FUNCTION nirs.protect_dataset_version();
CREATE FUNCTION nirs.protect_observation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP='DELETE' THEN RAISE EXCEPTION 'Observation is immutable'; END IF;
    IF NEW.sql_text IS DISTINCT FROM OLD.sql_text OR NEW.context_json IS DISTINCT FROM OLD.context_json
       OR NEW.unit_kind IS DISTINCT FROM OLD.unit_kind OR NEW.purpose IS DISTINCT FROM OLD.purpose
       OR NEW.sql_sha256 IS DISTINCT FROM OLD.sql_sha256
    THEN RAISE EXCEPTION 'Create a new observation for changed SQL or context'; END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER observation_content_immutable BEFORE UPDATE OR DELETE ON nirs.observations
    FOR EACH ROW EXECUTE FUNCTION nirs.protect_observation();
