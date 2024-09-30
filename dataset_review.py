import datasets

dataset = datasets.load_dataset('BaSalam/bslm-product-entity-cls-610k')

print(dataset)
# > DatasetDict({
# >     train: Dataset({
# >         features: ['title', 'description', 'entity'],
# >         num_rows: 609620
# >     })
# > })


print(len(dataset['train'].unique('entity')))
# > 61662
