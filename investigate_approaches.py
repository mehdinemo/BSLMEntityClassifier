import json
import os
from datetime import datetime

import datasets
from groq import Groq
from openai import OpenAI
from pydantic import BaseModel

from utils import logger_function

logger = logger_function(name="investigate_approaches")

model = "llama-3.2-11b-text-preview"  # gpt-4o-mini | llama-3.2-11b-text-preview
test_dataset_path = "test_ds.json"

logger.info(f"Starting investigate_approaches model: {model}")

if "gpt" in model:
    client = OpenAI()
else:
    client = Groq()


class Product(BaseModel):
    entity: str
    # attributes: Dict[str, str]


system_content_template = """Extract the main entity from the product title.

Response format:
```json
{output_schema}
```"""

# در مواردی که مدل داده پیچیده است از اسکیما استفاده کنیم
# output_schema = json.dumps(Product.model_json_schema())

# در این مورد چون تنها یه پارامتر میخایم و وراثت و ساختار پیچیده ای در کار نیست،
# و با هدف کاهش توکن ورودی یه ساختار خیلی ساده برای اسکیمای خروجی تعین کردیم
output_schema = '{"entity": "extracted entity"}'

system_content = system_content_template.format(output_schema=output_schema)


def zero_shot(user_content: str):
    completion_kw = dict(
        model=model,
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
        temperature=0
    )

    if "gpt" in model:
        completion_kw["response_format"] = {"type": "json_object"}

    try:
        st = datetime.now()
        completion = client.chat.completions.create(**completion_kw)
        duration = (datetime.now() - st).total_seconds()
    except:
        logger.exception("create completion")
        return None, None

    usage = completion.usage.to_dict()
    usage.update({"duration": duration})
    logger.info(f"successful completion: {usage}")

    try:
        product = Product.model_validate_json(completion.choices[0].message.content)
    except:
        logger.exception(f"ValidationError for completion content: {completion.choices[0].message.content}")
        return None, usage

    return product.entity, usage


def add_user_content(products):
    """
    اضافه کردن پیام کاربر با فرمت YAML به دیتاست برای کوئری به LLM
    استفاده نشده چون برای شروع فقط title رو استفاده میکنیم.
    """

    products["user_content"] = f"Product:\n- title: {products['title']}\n- description: {products['description']}"
    return products


def create_test_ds():
    """
    ساخت دیتاست تست با تعداد کم برای تست LLM ها
    مهمه که مدل ها روی یک دیتاست یکسان تست بشن تا پارامترهای مختلفشون رو مقایسه کنیم.
    چون در توسعه کد ممکنه مدل های مختلف تست بشن و شرایط مختلفی تاثیر گذاره ترجیحا این دیتاست تست رو روی دیسک ذخیره میکنیم.
    """

    dataset = datasets.load_dataset('BaSalam/bslm-product-entity-cls-610k')
    dataset = dataset["train"].train_test_split(test_size=20, shuffle=True)
    # ds["train"] = ds["train"].map(add_user_content, batched=True, batch_size=100)
    return dataset['test'].to_list()


def main():
    if not os.path.isfile(test_dataset_path):
        test_ds = create_test_ds()
        with open(test_dataset_path, 'w') as f:
            json.dump(test_ds, f, ensure_ascii=False, indent=2)
    else:
        with open(test_dataset_path, "r") as f:
            test_ds = json.load(f)

    extracted_entities = []
    for product in test_ds:
        extracted_entity, usage = zero_shot(product['title'])
        product.update(
            {
                "extracted_entity": extracted_entity,
                "usage": usage
            }
        )
        extracted_entities.append(product)

    with open(f'{model}.json', 'w') as f:
        json.dump(extracted_entities, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
