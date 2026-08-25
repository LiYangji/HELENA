def test_frozen_root_public_api_imports():
    import helena

    expected = (
        "FitResult",
        "ForwardResult",
        "ModelConfig",
        "OneZoneModel",
        "SEDDataset",
        "configure",
        "fit",
    )
    assert tuple(helena.__all__) == expected
    assert all(getattr(helena, name) is not None for name in expected)
