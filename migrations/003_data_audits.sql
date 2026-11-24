CREATE TABLE nirs.data_audits (
    audit_id text PRIMARY KEY,
    dataset_version_id text NOT NULL REFERENCES nirs.dataset_versions,
    audit_version text NOT NULL,
    configuration jsonb NOT NULL,
    summary jsonb NOT NULL,
    UNIQUE (audit_id, dataset_version_id)
);
CREATE TABLE nirs.observation_audits (
    audit_id text NOT NULL,
    dataset_version_id text NOT NULL,
    observation_id text NOT NULL,
    diagnostic jsonb NOT NULL,
    PRIMARY KEY (audit_id, observation_id),
    FOREIGN KEY (audit_id,dataset_version_id) REFERENCES nirs.data_audits(audit_id,dataset_version_id),
    FOREIGN KEY (dataset_version_id,observation_id) REFERENCES nirs.dataset_members(dataset_version_id,observation_id)
);
CREATE TABLE nirs.source_candidates (
    candidate_id text NOT NULL,
    registry_version text NOT NULL,
    metadata jsonb NOT NULL,
    PRIMARY KEY (candidate_id, registry_version)
);
CREATE TRIGGER data_audits_immutable BEFORE UPDATE OR DELETE ON nirs.data_audits
    FOR EACH ROW EXECUTE FUNCTION nirs.reject_change();
CREATE TRIGGER observation_audits_immutable BEFORE UPDATE OR DELETE ON nirs.observation_audits
    FOR EACH ROW EXECUTE FUNCTION nirs.reject_change();
CREATE TRIGGER source_candidates_immutable BEFORE UPDATE OR DELETE ON nirs.source_candidates
    FOR EACH ROW EXECUTE FUNCTION nirs.reject_change();
