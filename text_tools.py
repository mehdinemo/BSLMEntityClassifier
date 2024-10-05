import re

stop_words = {
    "generic_words": [
        'با',
        'و',
        'یا',
        'از',
        'برای',
        'تا',
        'به',
        'در',
        'این',
        'آن',
        'ای',
        'های',
        'ها'
    ],
    "generic_descriptors": [
        'مدل',
        'نوع',
        'طرح',
        'سایز',
        'رنگ',
        'جنس',
        'درجه',
        'محصول',
        'کد',
        'نمونه',
    ],
    "quantifiers_and_units": [
        'عدد',
        'عددی',
        'گرمی',
        'گرم'
        'کیلوگرمی',
        'کیلوگرم',
        'بسته',
        'تکه',
        'میلی‌لیتری',
        'لیتری',
        'لیتر',
        'میلی',
        'کیلویی',
        'کیلوی',
        'کیلو',
        'سانتی',
        'سانت',
        'متری',
        'متر',
        'هزار',
        'میلیون',
        'درصد',
        'بسیار'
    ],
    "generic_item_types": [
        'کالا',
        'محصول',
        'برند',
        'شرکت',
        'فروشگاه',
        'لوازم',
    ],
    "other": [
        'جدید',
        'اصالت',
        'عالی',
        'تضمین',
        'ارسال',
        'کیفیت'
    ]
}

all_stop_words = [
    stopword
    for stopwords_li in stop_words.values()
    for stopword in stopwords_li
]

chars = [
    '!',
    '"',
    '#',
    '(',
    ')',
    '*',
    ',',
    '-',
    '.',
    '/',
    ':',
    '[',
    ']',
    '«',
    '»',
    '،',
    '؛',
    '؟',
    '۰',
    '۱',
    '۲',
    '۳',
    '۴',
    '۵',
    '۶',
    '۷',
    '۸',
    '۹',
    '…',
    '$',
    'ْ',
    'ٌ',
    'ٍ',
    'ً',
    'ُ',
    'ِ',
    'َ',
    'ّ',
    'ء',
    'ٔ',
    'ٰ',
    '﷼'
]

split_chars = chars
split_chars.append(' ')
pattern = '[' + ''.join(re.escape(char) for char in split_chars) + ']'


def remove_stop_words(text: str) -> str:
    words = re.split(pattern, text)
    return ' '.join([word for word in words if word not in all_stop_words and word != ''])
