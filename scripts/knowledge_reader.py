import os
import json

class KnowledgeReader(object):
    def __init__(self, dataroot, knowledge_file):
        path = os.path.join(os.path.abspath(dataroot))

        with open(os.path.join(path, knowledge_file), 'r') as f:
            self.knowledge = json.load(f)

    def get_domain_list(self):
        return list(self.knowledge.keys())

    def get_entity_list(self, domain):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name")

        entity_ids = []
        for entity_id in self.knowledge[domain].keys():
            try:
                entity_id = int(entity_id)
                entity_ids.append(int(entity_id))
            except:
                pass

        result = []
        for entity_id in sorted(entity_ids):
            entity_name = self.knowledge[domain][str(entity_id)]['name']
            result.append({'id': entity_id, 'name': entity_name})

        return result

    def get_entity_name(self, domain, entity_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        result = self.knowledge[domain][str(entity_id)]['name'] or None

        return result


    def get_doc_list(self, domain=None, entity_id=None):
        if domain is None:
            domain_list = self.get_domain_list()
        else:
            if domain not in self.get_domain_list():
                raise ValueError("invalid domain name: %s" % domain)
            domain_list = [domain]

        result = []
        for domain in domain_list:
            if entity_id is None:
                for item_id, item_obj in self.knowledge[domain].items():
                    item_name = self.get_entity_name(domain, item_id)
                    
                    if item_id != '*':
                        item_id = int(item_id)

                    for doc_id, doc_obj in item_obj['docs'].items():
                        result.append({'domain': domain, 'entity_id': item_id, 'entity_name': item_name, 'doc_id': doc_id, 'doc': {'title': doc_obj['title'], 'body': doc_obj['body']}})
            else:
                if str(entity_id) not in self.knowledge[domain]:
                    raise ValueError("invalid entity id: %s" % str(entity_id))

                entity_name = self.get_entity_name(domain, entity_id)
                
                entity_obj = self.knowledge[domain][str(entity_id)]
                for doc_id, doc_obj in entity_obj['docs'].items():
                    result.append({'domain': domain, 'entity_id': entity_id, 'entity_name': entity_name, 'doc_id': doc_id, 'doc': {'title': doc_obj['title'], 'body': doc_obj['body']}})
        return result

    def get_doc(self, domain, entity_id, doc_id):
        if domain not in self.get_domain_list():
            raise ValueError("invalid domain name: %s" % domain)

        if str(entity_id) not in self.knowledge[domain]:
            raise ValueError("invalid entity id: %s" % str(entity_id))

        entity_name = self.get_entity_name(domain, entity_id)

        if str(doc_id) not in self.knowledge[domain][str(entity_id)]['docs']:
            raise ValueError("invalid doc id: %s" % str(doc_id))

        doc_obj = self.knowledge[domain][str(entity_id)]['docs'][str(doc_id)]
        result = {'domain': domain, 'entity_id': entity_id, 'entity_name': entity_name, 'doc_id': doc_id, 'doc': {'title': doc_obj['title'], 'body': doc_obj['body']}}

        return result
