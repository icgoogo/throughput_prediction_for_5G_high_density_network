def test_imports():
    import hdd_pipeline
    from hdd_pipeline.config import PipelineConfig
    assert PipelineConfig().target == "Throughput"
