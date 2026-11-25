-- A seal freezes both the split specification and its member assignments.
-- Historical split rows keep their original identities and schemas.
CREATE TABLE nirs.split_seals (
    split_id text PRIMARY KEY REFERENCES nirs.splits,
    member_count bigint NOT NULL CHECK (member_count >= 0),
    membership_sha256 text NOT NULL CHECK (membership_sha256 ~ '^[0-9a-f]{64}$'),
    specification_sha256 text NOT NULL CHECK (specification_sha256 ~ '^[0-9a-f]{64}$')
);
CREATE TRIGGER split_seals_immutable BEFORE UPDATE OR DELETE ON nirs.split_seals
    FOR EACH ROW EXECUTE FUNCTION nirs.reject_change();

CREATE FUNCTION nirs.seal_split(target_id text) RETURNS void LANGUAGE plpgsql AS $$
DECLARE cfg jsonb; n bigint; members_hash text; cfg_hash text; existing nirs.split_seals%ROWTYPE;
BEGIN
    SELECT specification INTO STRICT cfg FROM nirs.splits WHERE split_id=target_id FOR UPDATE;
    SELECT count(*), encode(sha256(convert_to(coalesce(string_agg(
        fold::text || ':' || part || ':' || observation_id, E'\n'
        ORDER BY fold,part COLLATE "C",observation_id COLLATE "C"),''),'UTF8')),'hex')
        INTO n,members_hash FROM nirs.split_members WHERE split_id=target_id;
    cfg_hash := encode(sha256(convert_to(cfg::text,'UTF8')),'hex');
    INSERT INTO nirs.split_seals VALUES (target_id,n,members_hash,cfg_hash) ON CONFLICT DO NOTHING;
    SELECT * INTO STRICT existing FROM nirs.split_seals WHERE split_id=target_id;
    IF existing.member_count<>n OR existing.membership_sha256<>members_hash
       OR existing.specification_sha256<>cfg_hash
    THEN RAISE EXCEPTION 'Frozen split checksum differs'; END IF;
END;
$$;

CREATE FUNCTION nirs.protect_split() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM nirs.split_seals WHERE split_id=OLD.split_id)
    THEN RAISE EXCEPTION 'Split is frozen'; END IF;
    IF TG_OP='DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
END;
$$;
CREATE TRIGGER frozen_split_specification BEFORE UPDATE OR DELETE ON nirs.splits
    FOR EACH ROW EXECUTE FUNCTION nirs.protect_split();

CREATE FUNCTION nirs.protect_split_member() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE target_id text;
BEGIN
    IF TG_OP='DELETE' THEN target_id:=OLD.split_id; ELSE target_id:=NEW.split_id; END IF;
    IF EXISTS (SELECT 1 FROM nirs.split_seals WHERE split_id=target_id)
       OR (TG_OP='UPDATE' AND EXISTS (SELECT 1 FROM nirs.split_seals WHERE split_id=OLD.split_id))
    THEN RAISE EXCEPTION 'Split membership is frozen'; END IF;
    IF TG_OP='DELETE' THEN RETURN OLD; ELSE RETURN NEW; END IF;
END;
$$;
CREATE TRIGGER frozen_split_members BEFORE INSERT OR UPDATE OR DELETE ON nirs.split_members
    FOR EACH ROW EXECUTE FUNCTION nirs.protect_split_member();

DO $$
DECLARE item record;
BEGIN
    FOR item IN SELECT split_id FROM nirs.splits LOOP
        PERFORM nirs.seal_split(item.split_id);
    END LOOP;
END;
$$;
