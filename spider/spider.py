import requests
import demjson
import csv

URL = "https://www.icourse163.org/web/j/mocCourseV2RpcBean.getCourseEvaluatePaginationByCourseIdOrTermId.rpc?csrfKey="
csrfKey = "aaa"
cookies = dict(NTESSTUDYSI=csrfKey)
session = requests.session()

courseId = 268001
pageIndex = 1
pageSize = 20
orderBy = 3

while True:
    print("crawling index" + str(pageIndex) + " page")
    result = session.post(URL + csrfKey, cookies=cookies,
                          data={'courseId': courseId, 'pageIndex': pageIndex, 'pageSize': pageSize, 'orderBy': orderBy})
    list_now = demjson.decode(result.content)['result']['list']
    csvWriter = csv.writer(open("output.csv", 'a', newline=''))
    if len(list_now) == 0:
        break
    for now in list_now:
        csvWriter.writerow([now['content'], now['mark']])
    pageIndex += 1
