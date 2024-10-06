import parser
import process

args = parser.parse_args()
print("#" * 64)
for i in vars(args):
    print(f"#{i:>40}: {str(getattr(args, i)):<20}#")
print("#" * 64)
process.main(args)