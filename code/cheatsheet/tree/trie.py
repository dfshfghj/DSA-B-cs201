class Node:
    def __init__(self, val=None):
        self.val = val
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, text):
        node = self.root
        has_prefix = False
        for word in text:
            if word not in node.children:
                node.children[word] = Node(word)
            node = node.children[word]
            if node.is_end:
                has_prefix = True
        node.is_end = True
        return has_prefix
    
if __name__ == '__main__':
    trie = Trie()
    trie.insert('apple')
    trie.insert('app')