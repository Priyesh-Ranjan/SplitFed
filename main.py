import variables
import process

if __name__ == "__main__":
    args = variables.parse_args()
    print("#" * 64)
    for i in vars(args):
        print(f"#{i:>40}: {str(getattr(args, i)):<20}#")
    print("#" * 64)
    process.main(args)
