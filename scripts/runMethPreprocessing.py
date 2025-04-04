# Build Method


def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # Remove mention

    text = re.sub(r'#[A-Za-z0-9]+', '', text) # Removing hashtags

    text = re.sub(r'RT[\s]', '', text) # delete RT

    text = re.sub(r"http\S+", '', text) # Delete links

    text = re.sub(r'[0-9]+', '', text) # Removing the numbers

    text = re.sub(r'[^\w\s]', '', text) # Remove characters other than letters and numbers


    text = text.replace('\n', ' ') # Replace a new line with a space

    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove all punctuation marks

    text = text.strip(' ') # Remove spaced characters from the left and right of text

    return text

def casefoldingText(text): # Convert all characters in text to lowercase

    text = text.lower()
    return text

def tokenizingText(text): # Split or split strings, text into token lists

    text = word_tokenize(text)
    return text

def filteringText(text): # Remove stopwords in text

    listStopwords = set(stopwords.words('indonesian'))
    listStopwords1 = set(stopwords.words('english'))
    listStopwords.update(listStopwords1)
    listStopwords.update(['iya','yaa','gak','nya','na','sih','ku',"di","ga","ya","gaa","loh","kah","woi","woii","woy"])
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text

def stemmingText(text): # Reducing a word to its basic form that removes prefix and suffix suffixes or to the root of a word
    # Create a stemmer object

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Break text into word lists

    words = text.split()

    # Apply a stemming to each word in the list

    stemmed_words = [stemmer.stem(word) for word in words]

    # Combining the words that have been voted on

    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text

def toSentence(list_words): # Turn a list of words into sentences

    sentence = ' '.join(word for word in list_words)
    return sentence

slangwords = ''
def fix_slangwords(text):
    words = text.split()
    fixed_words = []

    for word in words:
        if word.lower() in slangwords:
            fixed_words.append(slangwords[word.lower()])
        else:
            fixed_words.append(word)

    fixed_text = ' '.join(fixed_words)
    return fixed_text