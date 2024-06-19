from duckduckgo_search import DDGS

ddgs = DDGS()

def get_info_from_network(question):
    result = ddgs.text(keywords=question, region='cn-zh', max_results=1)
    info = result[0]['body']
    return info

if __name__ == '__main__':
    print(get_info_from_network('马保国是谁'))