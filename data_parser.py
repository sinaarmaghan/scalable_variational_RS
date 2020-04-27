import json
import numpy as np


def load_matrices(data, nonevalue):
    skills_hashtable = {'HTML': 0, 'CSS': 1, 'JS': 2, 'Liferay': 3, 'Java': 4
        , 'SQL': 5, 'JSP': 6, 'RE': 7, 'PM': 8, 'Spring': 9, 'Infrastructure': 10,
                        'Atlassian': 11, 'ServiceManagement': 12, 'SwArch': 13, 'JSF': 14, '.Net': 15,
                        'Selenium': 16, 'Neo4j': 17, 'Docker': 18, 'ELK': 19, 'iOS': 20, 'Talend': 21}
    my_cons_hastable = {}
    my_prj_hashtable = {}
    consultants = data['consultants']
    projects = data['projects']

    all_consult = find_all_consultants(consultants)
    active_consult = find_active_consultants(projects)
    inactive_consult = set(all_consult) - set(active_consult)

    '''
    print("\n\nFollowing consultants do not have a project!:\n")
    print(inactive_consult.__len__())
    print(inactive_consult)
    '''

    for i in range(active_consult.__len__()):
        my_cons_hastable.update({i: str(active_consult[i])})

    i = 0
    for key, value in projects.items():
        my_prj_hashtable.update({i: key})
        i = i + 1

    i = active_consult.__len__()
    j = projects.__len__()
    m = skills_hashtable.__len__() -1 #talend
    n = m

    x = np.full(shape=(i, j), fill_value=nonevalue, dtype='float128')
    f = np.full(shape=(m, i), fill_value=nonevalue, dtype='float128')
    g = np.full(shape=(n, j), fill_value=nonevalue, dtype='float128')

    for key, value in consultants.items():
        if int(key) in active_consult:

            i_place = get_key(key, my_cons_hastable)

            skill = value['skills']

            for technology, score in skill.items():
                if technology !="Talend":
                    f[skills_hashtable.get(technology), i_place] = score

    for key, value in projects.items():

        j_place = get_key(key, my_prj_hashtable)

        for k, v in value.items():
            if k == 'consultants':
                for cons in v:
                    x[get_key(cons, my_cons_hastable), j_place] = value['rentability']

            elif k == 'skillsrequired':
                for skill in v:
                    g[skills_hashtable.get(skill), j_place] = 1.0

    return x, g, f, my_cons_hastable, my_prj_hashtable


def generate_real_data(nonevalue):
    counter = 0
    with open('final_results2.json') as f:
        data = json.load(f)

    x, g, f, consultants_table, projects_table = load_matrices(data, nonevalue)

    for j in range(90):
        if np.isnan(x[:, j]).all():
            print("\nProjects without any consul ", projects_table.get(j))
        if np.isnan(g[:, j]).all():
            print("\n\nProjekt without any skill !", projects_table.get(j), "\n")

    for i in range(69):

        if np.isnan(x[i, :]).all():
            print(counter, "Consultat without project ", consultants_table.get(i))
            counter = counter + 1

    return x, g, f


def find_all_consultants(consultants):
    cons = set([])
    for key, value in consultants.items():
        cons.add(int(key))
    return sorted(cons)


def find_active_consultants(projects):
    active_consult = set([])
    for key, value in projects.items():
        for k, v in value.items():
            if k == 'consultants':
                for cons in v:
                    active_consult.add(int(cons))
    return sorted(active_consult)


def get_key(val, hashtable):
    for key, value in hashtable.items():
        if val == value:
            return key

    return "key doesn't exist"


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

