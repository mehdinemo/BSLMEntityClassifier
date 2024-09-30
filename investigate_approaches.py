import json
from datetime import datetime

import datasets
from pydantic import BaseModel

from openai import OpenAI
from utils import logger_function

logger = logger_function(name="investigate_approaches")

client = OpenAI()


class Product(BaseModel):
    entity: str
    # attributes: Dict[str, str]


system_content_template = """Extract the main entity from the product title.

Response format:
```json
{output_schema}
```"""

# output_schema = json.dumps(Product.model_json_schema())
output_schema = '{"entity": "extracted entity"}'

system_content = system_content_template.format(output_schema=output_schema)


def zero_shot(user_content: str) -> str | None:
    try:
        st = datetime.now()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        duration = (datetime.now() - st).total_seconds()
        usage = completion.usage.to_dict()
        usage.pop("completion_tokens_details")
        usage.update({"duration": duration})
        logger.info(f"successful completion: {usage}")
    except:
        logger.exception("create completion")
        return None

    try:
        product = Product.model_validate_json(completion.choices[0].message.content)
    except:
        logger.exception(f"ValidationError for completion content: {completion.choices[0].message.content}")
        return None

    return product.entity


def add_user_content(products):
    products["user_content"] = f"Product:\n- title: {products['title']}\n- description: {products['description']}"
    return products


def main():
    dataset = datasets.load_dataset('BaSalam/bslm-product-entity-cls-610k')

    dataset = dataset["train"].train_test_split(train_size=5, shuffle=True)

    # ds["train"] = ds["train"].map(add_user_content, batched=True, batch_size=100)

    extracted_entities = []
    for product in dataset["train"]:
        product.update(
            {
                "extracted_entity": zero_shot(product['title'])
            }
        )
        extracted_entities.append(product)

    with open('openai_res.json', 'w') as f:
        json.dump(extracted_entities, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
