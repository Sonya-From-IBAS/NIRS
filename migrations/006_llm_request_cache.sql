CREATE TABLE nirs.llm_pilots (
    pilot_id text PRIMARY KEY,
    configuration jsonb NOT NULL,
    max_api_attempts integer NOT NULL CHECK (max_api_attempts>0),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE nirs.llm_requests (
    request_key text PRIMARY KEY,
    origin_pilot_id text NOT NULL REFERENCES nirs.llm_pilots,
    specification jsonb NOT NULL,
    state text NOT NULL CHECK (state IN ('in_flight','complete','interrupted')),
    api_attempts smallint NOT NULL CHECK (api_attempts=1),
    request_artifact text NOT NULL REFERENCES nirs.artifacts,
    response_artifact text REFERENCES nirs.artifacts,
    result jsonb,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    CHECK ((state='complete' AND result IS NOT NULL AND response_artifact IS NOT NULL AND finished_at IS NOT NULL)
           OR (state<>'complete' AND result IS NULL AND response_artifact IS NULL))
);
CREATE FUNCTION nirs.check_llm_request() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE ceiling integer; used integer;
BEGIN
    IF TG_OP='INSERT' THEN
        SELECT max_api_attempts INTO STRICT ceiling FROM nirs.llm_pilots
            WHERE pilot_id=NEW.origin_pilot_id FOR UPDATE;
        SELECT COALESCE(sum(api_attempts),0) INTO used FROM nirs.llm_requests
            WHERE origin_pilot_id=NEW.origin_pilot_id;
        IF used+NEW.api_attempts>ceiling THEN RAISE EXCEPTION 'LLM pilot API budget exhausted'; END IF;
    ELSE
        IF OLD.state<>'in_flight' OR NEW.request_key<>OLD.request_key
           OR NEW.origin_pilot_id<>OLD.origin_pilot_id OR NEW.specification<>OLD.specification
           OR NEW.api_attempts<>OLD.api_attempts OR NEW.request_artifact<>OLD.request_artifact
        THEN RAISE EXCEPTION 'LLM request identity and finalized responses are immutable'; END IF;
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER llm_request_guard BEFORE INSERT OR UPDATE ON nirs.llm_requests
    FOR EACH ROW EXECUTE FUNCTION nirs.check_llm_request();
CREATE TRIGGER llm_request_no_delete BEFORE DELETE ON nirs.llm_requests
    FOR EACH ROW EXECUTE FUNCTION nirs.reject_change();
CREATE TRIGGER llm_pilot_immutable BEFORE UPDATE OR DELETE ON nirs.llm_pilots
    FOR EACH ROW EXECUTE FUNCTION nirs.reject_change();
