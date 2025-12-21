import sys
import pkg_resources

def check_package(name):
    try:
        ver = pkg_resources.get_distribution(name).version
        print(f"✅ {name}: {ver}")
    except:
        print(f"❌ {name}: Not found")

print("🔍 Checking Environment Versions...")
print(f"Python: {sys.version.split()[0]}")
check_package("transformers")
check_package("sentence-transformers")
check_package("torch")
check_package("accelerate")

print("\n🔍 Testing Import...")
try:
    from transformers import GenerationMixin
    print("✅ Successfully imported GenerationMixin")
except ImportError as e:
    print(f"❌ Import Failed: {e}")
    try:
        import transformers
        print(f"   transformers file: {transformers.__file__}")
    except: pass
