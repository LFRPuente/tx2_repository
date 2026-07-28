ALTER TABLE event_asset
    DROP CONSTRAINT event_asset_asset_type_check;

ALTER TABLE event_asset
    ADD CONSTRAINT event_asset_asset_type_check
    CHECK (asset_type IN (
        'video',
        'raw_video',
        'sidecar',
        'original_overlay',
        'rectified_overlay'
    ));
