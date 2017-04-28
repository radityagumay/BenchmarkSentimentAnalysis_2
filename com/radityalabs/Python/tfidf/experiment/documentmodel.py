
class documentmodel:
    def getDocuments(self):
        document_tokens1 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
        document_tokens2 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
        document_tokens3 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
        document_tokens4 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
        document_tokens5 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
        document_tokens6 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
        document_tokens7 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

        documents = []
        documents.append(document_tokens1)
        documents.append(document_tokens2)
        documents.append(document_tokens3)
        documents.append(document_tokens4)
        documents.append(document_tokens5)
        documents.append(document_tokens6)
        documents.append(document_tokens7)
        return documents

