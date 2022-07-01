from algoliasearch.search_client import SearchClient

client = SearchClient.create(
    'KZGSXGDH9X',
    'f5d07b54762493d587052d741873d227'
)

index = client.init_index('demo_ecommerce')

index.set_settings({
    # Select the attributes you want to search in
    'searchableAttributes': [
        'brand', 'name', 'categories', 'description'
    ],
    # Define business metrics for ranking and sorting
    'customRanking': [
        'desc(popularity)'
    ],
    # Set up some attributes to filter results on
    'attributesForFaceting': [
        'categories', 'searchable(brand)', 'price'
    ]
})
