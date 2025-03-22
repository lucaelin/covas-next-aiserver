import traceback

if __name__ == "__main__":

    try:
        import AIServerInternal
        AIServerInternal.main()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print("Press Enter to quit...")
        input()
        exit(1)