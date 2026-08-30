import os
import gzip
import tarfile
import tempfile
import urllib.request
import shutil

# Define ArXiv paper ID
paper_id = "2608.27454"
# paper_id = "2608.26247"
# paper_id = "2608.21267"

def download_pdf(paper_id, output_dir="."):
  os.makedirs(output_dir, exist_ok=True)
  url = f"https://arxiv.org/pdf/{paper_id}.pdf"
  output_file = os.path.join(output_dir, f"{paper_id}.pdf")

  urllib.request.urlretrieve(url, output_file)

  print(f"Downloaded PDF: {output_file}")


def download_tex(paper_id, output_dir="."):
    url = f"https://arxiv.org/e-print/{paper_id}"

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Download the source file
        urllib.request.urlretrieve(url, tmp_path)

        # Check if it is a tar archive
        if tarfile.is_tarfile(tmp_path):
            with tarfile.open(tmp_path, "r:*") as tar:
                tar.extractall(output_dir)

        # ---------------------------------
        # Filter to only largest .tex file
        tex_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(output_dir)
            for file in files
            if file.endswith(".tex")
        ]

        if tex_files:
            # Find largest .tex file
            largest_tex = max(tex_files, key=os.path.getsize)

            # Rename it to paper_id.tex
            final_path = os.path.join(output_dir, f"{paper_id}.tex")
            os.replace(largest_tex, final_path)

            # Delete everything else in the folder
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)

                if item_path != final_path:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)

            print(f"Downloaded TeX: {final_path}")

        # ---------------------------------

        else:
            # Otherwise, assume it is a gzip-compressed TeX file
            output_file = os.path.join(output_dir, f"{paper_id}.tex")

            with gzip.open(tmp_path, "rb") as f_in:
                with open(output_file, "wb") as f_out:
                    f_out.write(f_in.read())

            print(f"Downloaded TeX: {output_file}")

    finally:
        os.remove(tmp_path)


def download_paper(tex_dir = None, pdf_dir = None):
    if tex_dir is None:
        tex_dir = paper_id
    if pdf_dir is None:
        pdf_dir = paper_id
    os.makedirs(tex_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
    tex_path = os.path.join(tex_dir, f"{paper_id}.tex")

    if not os.path.exists(tex_path):
        print("TeX not found. Downloading...")
        download_tex(paper_id, output_dir=tex_dir)
    else:
        print("TeX already exists. Skipping download.")
    # Note download_tex deletes all other files in the folder to keep only largest tex file

    if not os.path.exists(pdf_path):
            print("PDF not found. Downloading...")
            download_pdf(paper_id, output_dir=pdf_dir)
    else:
        print("PDF already exists. Skipping download.")


download_paper(tex_dir = paper_id, pdf_dir = paper_id)