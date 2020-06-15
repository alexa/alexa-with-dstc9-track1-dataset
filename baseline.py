import sys

if __name__ == "__main__":
    if '--generate' in sys.argv:
        from baseline import generate as main
    else:
        from baseline import main

    main.main()
