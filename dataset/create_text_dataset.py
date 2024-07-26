import requests
from bs4 import BeautifulSoup

def fetch_mayo_clinic_page(url):
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Fetch the title
    title = soup.find('h1').get_text()
    print("Page Title: ", title)
    
    # Fetch the main content
    content = soup.find_all('div', {'class': 'content'})
    for section in content:
        print("\nSection Content:\n", section.get_text(strip=True), "...")
        
        
        # Uncomment the line below to print the entire section content (may be very long)
        # print("\nSection Content:\n", section.get_text(strip=True))

if __name__ == "__main__":
    fetch_mayo_clinic_page("https://eyewiki.aao.org/Incision_Construction")
