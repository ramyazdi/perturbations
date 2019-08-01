__author__ = 'max'


class Sentence(object):
    def __init__(self, words, word_ids):
        self.words = words
        self.word_ids = word_ids

    def length(self):
        return len(self.words)


class DependencyInstance(object):
    def __init__(self, sentence, postags, pos_ids, heads, arctags, arc_ids):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.heads = heads
        self.arctags = arctags
        self.arc_ids = arc_ids

    def length(self):
        return self.sentence.length()


class NERInstance(object):
    def __init__(self, sentence, postags, pos_ids, chunk_tags, chunk_ids, ner_tags, ner_ids):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.chunk_tags = chunk_tags
        self.chunk_ids = chunk_ids
        self.ner_tags = ner_tags
        self.ner_ids = ner_ids

    def length(self):
        return self.sentence.length()