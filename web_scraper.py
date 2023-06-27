import requests
from bs4 import BeautifulSoup
import re


def scrape_multiple_pages(page_urls):
    """
    Scrapes data from multiple pages.

    Args:
        page_urls (list): A list of URLs of the pages to scrape.

    Returns:
        list: A list of scraped data from each page.
    """
    scraped_data = []
    for url in page_urls:
        try:
            with WebScraper(url) as scraper:
                # Perform scraping on each page
                data = scraper.scrape()
                scraped_data.append(data)
        except RuntimeError as e:
            print(f"Error scraping page '{url}': {e}")

    return scraped_data


class WebScraper:
    def __init__(self, url):
        self.url = url

    def __enter__(self):
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            self.soup = BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch the URL: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup code
        self.soup = None

    def fetch_title(self):
        """
        Fetches the page title from the HTML content.
        Returns:
            str: The page title, or None if not found.
        """
        try:
            if self.soup:
                title_tag = self.soup.title
                if title_tag:
                    title = title_tag.text.strip()
                    return title
        except AttributeError:
            pass

        return None

    def fetch_links(self):
        """
        Fetches all the links from the HTML content.

        Returns:
            list: A list of links as strings.
        """
        try:
            if self.soup:
                links = self.soup.find_all('a')
                return [link['href'] for link in links]
        except (AttributeError, KeyError):
            pass

        return []

    def scrape(self):
        """
        Scrapes the page title and links.

        Returns:
            dict: A dictionary with 'title' and 'links' keys.
        """
        title = self.fetch_title()
        links = self.fetch_links()

        return {'title': title, 'links': links}

    @property
    def title(self):
        """
        Property that provides direct access to the page title.

        Returns:
            str: The page title.
        """
        return self.fetch_title()

    @property
    def links(self):
        """
        Property that provides direct access to the links on the page.

        Returns:
            list: A list of links as strings.
        """
        return self.fetch_links()

    def find_elements(self, element_type, **kwargs):
        """
        Finds elements of a specific type with optional attributes.

        Args:
            element_type (str): The type of element to search for (e.g., 'p', 'div').
            **kwargs: Optional attributes to filter the elements.

        Returns:
            list: A list of matching elements.
        """
        try:
            if self.soup:
                elements = self.soup.find_all(element_type, **kwargs)
                return elements
        except (AttributeError, KeyError):
            pass

        return []

    def find_element(self, element_type, **kwargs):
        """
        Finds the first element of a specific type with optional attributes.

        Args:
            element_type (str): The type of element to search for (e.g., 'p', 'div').
            **kwargs: Optional attributes to filter the elements.

        Returns:
            bs4.element.Tag: The first matching element, or None if not found.
        """
        try:
            if self.soup:
                element = self.soup.find(element_type, **kwargs)
                return element
        except (AttributeError, KeyError):
            pass

        return None

    def count_elements(self, element_type, **kwargs):
        """
        Counts the number of elements of a specific type with optional attributes.

        Args:
            element_type (str): The type of element to count (e.g., 'p', 'div').
            **kwargs: Optional attributes to filter the elements.

        Returns:
            int: The count of matching elements.
        """
        try:
            if self.soup:
                elements = self.soup.find_all(element_type, **kwargs)
                return len(elements)
        except (AttributeError, KeyError):
            pass

        return 0

    def has_element(self, element_type, **kwargs):
        """
        Checks if at least one element of a specific type with optional attributes exists.

        Args:
            element_type (str): The type of element to check (e.g., 'p', 'div').
            **kwargs: Optional attributes to filter the elements.

        Returns:
            bool: True if at least one matching element exists, False otherwise.
        """
        try:
            if self.soup:
                element = self.soup.find(element_type, **kwargs)
                return element is not None
        except (AttributeError, KeyError):
            pass

        return False


    def get_element_text(self, element_type, **kwargs):
        """
        Retrieves the text content of the first element of a specific type with optional attributes.

        Args:
            element_type (str): The type of element to search for (e.g., 'p', 'div').
            **kwargs: Optional attributes to filter the elements.

        Returns:
            str: The text content of the first matching element, or None if not found.
        """
        try:
            if self.soup:
                element = self.soup.find(element_type, **kwargs)
                if element:
                    return element.text.strip()
        except (AttributeError, KeyError):
            pass

        return None

    def get_element_attribute(self, element_type, attribute, **kwargs):
        """
        Retrieves the value of a specific attribute from the first element of a specific type with optional attributes.

        Args:
            element_type (str): The type of element to search for (e.g., 'p', 'div').
            attribute (str): The attribute name to retrieve the value from.
            **kwargs: Optional attributes to filter the elements.

        Returns:
            str: The value of the specified attribute from the first matching element, or None if not found.
        """
        try:
            if self.soup:
                element = self.soup.find(element_type, **kwargs)
                if element and attribute in element.attrs:
                    return element[attribute]
        except (AttributeError, KeyError):
            pass

        return None

    def find_element_by_text(self, element_type, text):
        """
        Finds the first element of a specific type that contains the given text.

        Args:
            element_type (str): The type of element to search for (e.g., 'p', 'div').
            text (str): The text to search for within the elements.

        Returns:
            bs4.element.Tag: The first matching element, or None if not found.
        """
        try:
            if self.soup:
                element = self.soup.find(element_type, text=text)
                return element
        except (AttributeError, KeyError):
            pass

        return None

    def fetch_images(self):
        """
        Fetches all the image URLs from the HTML content.

        Returns:
            list: A list of image URLs.
        """
        try:
            if self.soup:
                images = self.soup.find_all('img')
                return [image['src'] for image in images]
        except (AttributeError, KeyError):
            pass

        return []

    def fetch_form_data(self):
        """
        Fetches the form data from the HTML content.

        Returns:
            dict: A dictionary containing form field names as keys and their values.
        """
        try:
            if self.soup:
                form = self.soup.find('form')
                if form:
                    form_data = {}
                    inputs = form.find_all('input')
                    for input_tag in inputs:
                        if 'name' in input_tag.attrs:
                            field_name = input_tag['name']
                            field_value = input_tag.get('value', '')
                            form_data[field_name] = field_value
                    return form_data
        except (AttributeError, KeyError):
            pass

        return {}

    def scrape_table(self, table_id):
        """
        Scrape data from an HTML table.

        Args:
            table_id (str): The ID of the table to scrape.

        Returns:
            list: A list of dictionaries representing each row of data in the table.
                  The keys are the table header column names and the values are the corresponding cell values.
        """
        try:
            if self.soup:
                table = self.soup.find('table', id=table_id)
                if table:
                    header_row = table.find('thead').find('tr')
                    header_cols = header_row.find_all('th')
                    headers = [header.text.strip() for header in header_cols]

                    data_rows = table.find('tbody').find_all('tr')
                    scraped_data = []
                    for row in data_rows:
                        cells = row.find_all('td')
                        row_data = {}
                        for idx, cell in enumerate(cells):
                            row_data[headers[idx]] = cell.text.strip()
                        scraped_data.append(row_data)

                    return scraped_data
        except (AttributeError, KeyError):
            pass

        return []

    def extract_text_with_regex(self, pattern):
        """
        Extracts text from the HTML content using a regular expression pattern.

        Args:
            pattern (str): The regular expression pattern to match against the HTML content.

        Returns:
            list: A list of extracted text that matches the pattern.
        """
        try:
            if self.soup:
                content = self.soup.get_text()
                matches = re.findall(pattern, content)
                return matches
        except (AttributeError, re.error):
            pass

        return []


# Example usage
url = 'https://example.com'

try:
    with WebScraper(url) as scraper:
        data = scraper.scrape()
        print('Title:', data['title'])
        print('Links:', data['links'])

        # Accessing title and links using properties
        print('Title (Property):', scraper.title)
        print('Links (Property):', scraper.links)

        # Finding specific elements
        paragraphs = scraper.find_elements('p')
        for p in paragraphs:
            print('Paragraph:', p.text)

        first_heading = scraper.find_element('h1')
        if first_heading:
            print('First Heading:', first_heading.text)

        # Counting elements
        num_divs = scraper.count_elements('div')
        print('Number of Divs:', num_divs)

        # Checking if an element exists
        has_img = scraper.has_element('img', src='example.jpg')
        print('Has Image:', has_img)

        # Retrieving text content of an element
        paragraph_text = scraper.get_element_text('p', class_='highlight')
        print('Paragraph Text:', paragraph_text)

        # Retrieving value of an attribute from an element
        image_source = scraper.get_element_attribute('img', 'src', alt='Example Image')
        print('Image Source:', image_source)

        # Finding an element by its text
        div_element = scraper.find_element_by_text('div', 'Example')
        if div_element:
            print('Found div element:', div_element)

        # Fetching image URLs
        images = scraper.fetch_images()
        print('Images:', images)

        # Fetching form data
        form_data = scraper.fetch_form_data()
        print('Form Data:', form_data)

        # Scraping table data
        table_data = scraper.scrape_table('my-table-id')
        for row in table_data:
            print(row)

except RuntimeError as e:
    print('Error:', e)
