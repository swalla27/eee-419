from googletrans import Translator

def english_to_swedish(input_text: str):
    try:
        translator = Translator()
        translated = translator.translate(input_text, src='en', dest='sv')
        return translated.text
    except Exception as e:
        return f'An error occurred: {e}'
    
def swedish_to_english(input_text: str):
    try:
        translator = Translator()
        translated = translator.translate(input_text, src='sv', dest='en')
        return translated.text
    except Exception as e:
        return f'An error occurred: {e}'

print(english_to_swedish('to paint'))
print(swedish_to_english('att älska'))