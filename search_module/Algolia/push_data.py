import requests
from algoliasearch.search_client import SearchClient

client = SearchClient.create(
    'KZGSXGDH9X',
    'f5d07b54762493d587052d741873d227'
)

index = client.init_index('demo_ecommerce')

products = requests.get(
    'https://alg.li/doc-ecommerce.json'
)

index.save_objects(products.json(), {
    'autoGenerateObjectIDIfNotExist': True
})
