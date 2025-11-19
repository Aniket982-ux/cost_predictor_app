import os

def test_lgbm_model_exists():
    assert os.path.exists("trained_lgbm_model.txt")

def test_refiner_model_exists():
    assert os.path.exists("embedding_refiner_checkpoint.pth")
