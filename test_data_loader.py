import os
import sys
import json
from unittest.mock import patch # For mocking CosmosDB if needed
from dotenv import load_dotenv # Import load_dotenv

# Add the parent directory to the Python path to allow importing data_loader
# This is crucial if test_data_loader.py is inside a 'tests' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import FileLoader, CosmosDBLoader, get_data_loader

# Load environment variables from .env file
load_dotenv()

def run_data_loader_tests():
    print("--- Testing data_loader.py ---")

    # --- Test FileLoader ---
    print("\n--- Testing FileLoader ---")
    
    # Create a dummy test file
    test_file_path = "test_document.txt"
    test_content = "This is a test document for FileLoader. It contains some sample text."
    with open(test_file_path, "w") as f:
        f.write(test_content)
    print(f"Created dummy file: {test_file_path}")

    try:
        file_loader = FileLoader(test_file_path)
        loaded_data = file_loader.load_data()
        # Only print the title for FileLoader
        if loaded_data and loaded_data[0].get("title"):
            print(f"Loaded data from file. Title: {loaded_data[0]['title']}")
        else:
            print("Loaded data from file (no title found).")

        assert len(loaded_data) == 1
        assert isinstance(loaded_data[0], dict)
        assert loaded_data[0].get("content") == test_content
        assert loaded_data[0].get("title") == os.path.basename(test_file_path)
        print("FileLoader test PASSED.")
    except FileNotFoundError as e:
        print(f"FileLoader test FAILED: {e}")
    except AssertionError:
        print("FileLoader test FAILED: Content mismatch.")
    except Exception as e:
        print(f"FileLoader test FAILED unexpectedly: {e}")
    finally:
        # Clean up the dummy file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            print(f"Cleaned up dummy file: {test_file_path}")

    # Test FileLoader with a non-existent file
    print("\n--- Testing FileLoader with non-existent file ---")
    try:
        non_existent_file = "non_existent_file.txt"
        FileLoader(non_existent_file)
        print("FileLoader non-existent file test FAILED: Expected FileNotFoundError.")
    except FileNotFoundError:
        print("FileLoader non-existent file test PASSED: Correctly raised FileNotFoundError.")
    except Exception as e:
        print(f"FileLoader non-existent file test FAILED unexpectedly: {e}")


    # --- Test CosmosDBLoader (using credentials from .env) ---
    print("\n--- Testing CosmosDBLoader (using credentials from .env) ---")
    
    # Get CosmosDB credentials from environment variables
    cosmosdb_conn_str = os.getenv("GRAPHRAG_COSMOSDB_CONN_STR")
    cosmosdb_db_name = os.getenv("GRAPHRAG_COSMOSDB_DB_NAME")
    cosmosdb_container_name = os.getenv("GRAPHRAG_COSMOSDB_CONTAINER_NAME")

    if not all([cosmosdb_conn_str, cosmosdb_db_name, cosmosdb_container_name]):
        print("Skipping CosmosDBLoader test: CosmosDB environment variables not set. "
              "Please set GRAPHRAG_COSMOSDB_CONN_STR, GRAPHRAG_COSMOSDB_DB_NAME, and GRAPHRAG_COSMOSDB_CONTAINER_NAME in your .env file.")
    else:
        try:
            cosmos_loader = CosmosDBLoader(
                connection_string=cosmosdb_conn_str,
                database_name=cosmosdb_db_name,
                container_name=cosmosdb_container_name
            )
            cosmos_data = cosmos_loader.load_data()
            # Removed the print of full cosmos_data here
            
            # Assertions now expect a list of dictionaries
            assert isinstance(cosmos_data, list)
            if cosmos_data: # Check if list is not empty before accessing elements
                assert isinstance(cosmos_data[0], dict)
                assert "title" in cosmos_data[0]
                assert "content" in cosmos_data[0]
            
            print("\nCosmosDB Document Titles:")
            if cosmos_data:
                for doc in cosmos_data:
                    print(f"- {doc.get('title', 'No Title')}")
            else:
                print("No documents loaded from CosmosDB.")

            print("CosmosDBLoader test PASSED (using live data).")
        except AssertionError:
            print("CosmosDBLoader test FAILED: Content mismatch or unexpected structure.")
        except Exception as e:
            print(f"CosmosDBLoader test FAILED unexpectedly: {e}")

    # --- Test get_data_loader factory function ---
    print("\n--- Testing get_data_loader factory function ---")

    # Test with 'file' source type
    print("\n--- Testing get_data_loader with 'file' source type ---")
    # Create dummy file for this test
    factory_test_file_path = "factory_test_doc.txt"
    factory_test_content = "Content for factory file loader test."
    with open(factory_test_file_path, "w") as f:
        f.write(factory_test_content)

    try:
        file_config = {"source_type": "file", "file_path": factory_test_file_path}
        loader_from_factory = get_data_loader(file_config)
        assert isinstance(loader_from_factory, FileLoader)
        factory_loaded_data = loader_from_factory.load_data()
        # Only print the title for factory FileLoader
        if factory_loaded_data and factory_loaded_data[0].get("title"):
            print(f"Loaded data from factory file. Title: {factory_loaded_data[0]['title']}")
        else:
            print("Loaded data from factory file (no title found).")

        assert len(factory_loaded_data) == 1
        assert isinstance(factory_loaded_data[0], dict)
        assert factory_loaded_data[0].get("content") == factory_test_content
        assert factory_loaded_data[0].get("title") == os.path.basename(factory_test_file_path)
        print("get_data_loader (file) test PASSED.")
    except AssertionError:
        print("get_data_loader (file) test FAILED: Incorrect loader type or content.")
    except Exception as e:
        print(f"get_data_loader (file) test FAILED unexpectedly: {e}")
    finally:
        if os.path.exists(factory_test_file_path):
            os.remove(factory_test_file_path)

    # Test with 'cosmosdb' source type
    print("\n--- Testing get_data_loader with 'cosmosdb' source type ---")
    # Use the same .env loaded credentials for the factory test
    cosmos_config_factory = {
        "source_type": "cosmosdb",
        "cosmosdb_connection_string": cosmosdb_conn_str,
        "cosmosdb_database_name": cosmosdb_db_name,
        "cosmosdb_container_name": cosmosdb_container_name
    }
    
    if not all([cosmosdb_conn_str, cosmosdb_db_name, cosmosdb_container_name]):
        print("Skipping get_data_loader (cosmosdb) test: CosmosDB environment variables not set.")
    else:
        try:
            loader_from_factory = get_data_loader(cosmos_config_factory)
            assert isinstance(loader_from_factory, CosmosDBLoader)
            factory_loaded_data = loader_from_factory.load_data()
            # Removed the print of full factory_loaded_data here
            
            # Assertions now expect a list of dictionaries
            assert isinstance(factory_loaded_data, list)
            if factory_loaded_data: # Check if list is not empty before accessing elements
                assert isinstance(factory_loaded_data[0], dict)
                assert "title" in factory_loaded_data[0]
                print("\nCosmosDB Document Titles (via factory):")
                for doc in factory_loaded_data:
                    print(f"- {doc.get('title', 'No Title')}")
                print("get_data_loader (cosmosdb) test PASSED (using live data).")
            else:
                print("No documents loaded from CosmosDB via factory.")
        except AssertionError:
            print("get_data_loader (cosmosdb) test FAILED: Incorrect loader type or content.")
        except Exception as e:
            print(f"get_data_loader (cosmosdb) test FAILED unexpectedly: {e}")

    # Test with unsupported source type
    print("\n--- Testing get_data_loader with unsupported source type ---")
    try:
        unsupported_config = {"source_type": "unsupported_type"}
        get_data_loader(unsupported_config)
        print("get_data_loader (unsupported) test FAILED: Expected ValueError.")
    except ValueError as e:
        print(f"get_data_loader (unsupported) test PASSED: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"get_data_loader (unsupported) test FAILED unexpectedly: {e}")

    print("\n--- All data_loader tests complete ---")

if __name__ == "__main__":
    run_data_loader_tests()
