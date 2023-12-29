import calibre_search as cali

def main():
    res =  cali.books['The Genesis of Good and Evil']

    print(res.tags)

if __name__ == "__main__":
    main()