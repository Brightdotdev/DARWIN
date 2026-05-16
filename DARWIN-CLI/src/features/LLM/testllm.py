


from src.features.LLM.LLMManager import LLMManager
from src.lib.KeyManager import load_keys_from_keychain


def test_manager():
    import time
    
    t0 = time.perf_counter()
    t1 = time.perf_counter()
    
    LLMManager.initialize()
    client = LLMManager.get_client()
    message = client.call_small([{"role": "user", "content": "What is a large language model?"}],)
    elapsed = time.perf_counter() - t0
    message2 = client.call_small([{"role": "user", "content": "What is a large language model?"}],)
    elapsed2 = time.perf_counter() - t1
    print(message)
    print(elapsed)
    print(message2)
    print(elapsed2)
    


if __name__ == "__main__":
    test_manager()