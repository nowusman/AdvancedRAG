import logging
import torch
import time
from PIL import Image
import tempfile
import fitz
import shutil
from bs4 import BeautifulSoup
import re
import os
import math

MAX_TOKEN = 2048

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    # from huggingface_hub import login
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument
    docling_available = True
except ImportError:
    docling_available = False


class PDFProcessor:
    def __init__(self, file_path, model_folder):
        self.upload_option = "PDF File"
        self.task_type = "Convert this page to docling."
        self.file_path = file_path
        self.model_folder = model_folder

    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        missing = []
        if not transformers_available:
            missing.append("transformers huggingface_hub")
        if not docling_available:
            missing.append("docling-core")
        return missing

    def process_single_image(self, image):
        """Process a single image"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        start_time = time.time()
        
        # Load processor and model (simplified to avoid potential issues)
        try:
            processor = AutoProcessor.from_pretrained(self.model_folder)
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_folder,
                torch_dtype=torch.float32,  # Using simpler dtype
            ).to(device)
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
        
        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.task_type}
                ]
            },
        ]
        
        # Prepare inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(device)
        
        # Generate outputs
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_TOKEN)
        prompt_length = inputs.input_ids.shape[1]
        trimmed_generated_ids = generated_ids[:, prompt_length:]
        doctags = processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=False,
        )[0].lstrip()
        
        # Clean the output
        doctags = doctags.replace("<end_of_utterance>", "").strip()

        ### Add end mark
        if "</doctag>" not in doctags:
            doctags += "</doctag>"
        
        # Populate document
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        
        # Create a docling document
        doc = DoclingDocument(name="Document")
        doc.load_from_doctags(doctags_doc)
        
        # Export as markdown
        md_content = doc.export_to_markdown()
        
        processing_time = time.time() - start_time
        
        return doctags, md_content, image, processing_time

    def process_pdf(self):
        """Process PDF using PyMuPDF (fitz) to extract text from images"""

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(open(self.file_path, 'rb'), temp_file)

        pdf_path = temp_file.name  # Get the path to the temp file

        # Open the PDF using PyMuPDF (fitz)
        doc = fitz.open(pdf_path)
        
        all_doctags = []
        all_md_content = []
        all_page_images = []
        total_processing_time = 0

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Convert page to image (pixmap)
            pix = page.get_pixmap()
            
            # Convert pixmap to PIL image
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            all_page_images.append(image)
            
            # Process the image using the same method as before
            doctags, md_content, img, processing_time = self.process_single_image(image)
            
            all_doctags.append(doctags)
            all_md_content.append(md_content)
            total_processing_time += processing_time
        
        # Combine all doctags and markdown content for the entire PDF
        combined_doctags = "\n\n".join(all_doctags)
        combined_md_content = "\n\n".join(all_md_content)
        
        return combined_doctags, combined_md_content, all_page_images, total_processing_time

    def smoldocling(self):
        '''
        upload_option: "Single Image", "Multiple Images", "PDF File"
        task_type: "Convert this page to docling.", "Convert this table to OTSL.", "Convert code to text.", "Convert formula to latex.", "Convert chart to OTSL.", "Extract all section header elements on the page."
        file_path: string, path of single image or pdf, or path list of multiple images
        model_folder: where llm is saved
        '''
        logging.basicConfig(level=logging.DEBUG)
        # Check dependencies
        missing_deps = self.check_dependencies()
        # logger = logging.getLogger(__name__)
        if missing_deps:
            logging.error(f"Missing dependencies: {', '.join(missing_deps)}. Please install them to use this app.")
            logging.info("Install with: pip install " + " ".join(missing_deps))

        
        if self.upload_option == "Single Image":
            if self.file_path is not None:
                try:
                    image = Image.open(self.file_path).convert("RGB")
                    doctags, md_content, img, processing_time = self.process_single_image(image)
                    logging.info(f"Processing completed in {processing_time:.2f} seconds")
                    return(doctags, md_content, img, processing_time)
                except Exception as e:
                    logging.error(f"Error processing image: {str(e)}")
            else:
                logging.error(f"No image.")
        elif self.upload_option == "Multiple Images":
            if self.file_path is not None:
                try:
                    images = [Image.open(file).convert("RGB") for file in self.file_path]
                    if len(images) > 0:
                        results = []
                        for idx, image in enumerate(images):
                            logging.info(f"Processing image {idx+1}/{len(images)}...")
                            doctags, md_content, img, processing_time = self.process_single_image(image)
                            results.append((doctags, md_content, img, processing_time))
                            logging.info(f"Image {idx+1} processed in {processing_time:.2f} seconds")
                        logging.info(f"All images processed successfully")
                        return(results)
                    else:
                        logging.error(f"No image.")
                except Exception as e:
                    logging.error(f"Error processing images: {str(e)}")
            else:
                logging.error(f"No image.")
        elif self.upload_option == "PDF File":
            try:
                combined_doctags, combined_md_content, all_page_images, total_processing_time = self.process_pdf()
                logging.info(f"PDF processed successfully in {total_processing_time:.2f} seconds")
                return(combined_doctags, combined_md_content, all_page_images, total_processing_time)
            except Exception as e:
                logging.error(f"Error processing PDF: {str(e)}")



class DoctagParser:
    def __init__(self, doctagstr, page_images, image_save_folder, pdf_name):
        self.doctagstr = doctagstr
        # self.soup = BeautifulSoup(doctagstr, "html.parser")
        self.page_images = page_images
        self.image_save_folder = image_save_folder
        self.pdf_name = pdf_name

    def clean_truncated_before_end(self, html):
        doctag_end = html.find("</doctag>")
        if doctag_end == -1:
            return html  # no closing doctag

        before_end = html[:doctag_end]
        after_end = html[doctag_end:]

        # Find last incomplete "<tag" without closing ">"
        truncated_match = re.search(r"<\w+[^>]*$", before_end)
        if truncated_match:
            before_end = before_end[:truncated_match.start()]

        return before_end + after_end

    def remove_dangling_tags(self, text):
        text_end = text.find("</doctag>")
        last_tag_end = text.rfind("\n")
        if text_end>(last_tag_end+1) and last_tag_end!=-1:
            last_tag = text[last_tag_end+1:text_end]
            first = last_tag.find("<")
            end = last_tag.find(">")
            tag_name = last_tag[first+1:end]
            if tag_name!="doctag":
                tag_complete = last_tag.find(f"</{tag_name}>")
                if tag_complete==-1:
                    text = text[:last_tag_end+1]+"</doctag>"
        return text
    
    def parse_image(self, img, page_id, img_id, default_grid_size=500, delta_percent=0.005, threshold=0.01):
        try:
            loc_tags = [tag.name for tag in img.find_all() if tag.name.startswith("loc_")]
            x0, y0, x1, y1 = [int(loc.split('loc_')[1]) for loc in loc_tags]
            page_image = self.page_images[page_id]
            width, height = page_image.size
            
            # use delta to relieve the inaccuracy of coordinates
            deltax = width*delta_percent
            deltay = height*delta_percent
            x0 = max(int((x0-deltax)/default_grid_size*width),0)            # x, y are normlized with default_grid_size 
            y0 = max(int((y0-deltay)/default_grid_size*height),0)
            x1 = min(math.ceil((x1+deltax)/default_grid_size*width), width)
            y1 = min(math.ceil((y1+deltay)/default_grid_size*height), height)

            if (x1-x0)*(y1-y0)/width/height>threshold:
                image = page_image.crop((x0, y0, x1, y1))
                image_save_path = os.path.join(self.image_save_folder, f"{self.pdf_name}_page_{page_id+1}_image{img_id+1}.png")
                image.save(image_save_path)
                return(True)
            else:
                return(False)
        except Exception as e:
            logging.error(f"Error parsing image for page {page_id+1}: {str(e)}")

    def parse_otsl(self, otsl_tag):
        try:
            otsl_html = str(otsl_tag)

            # Extract caption if present
            caption_match = re.search(r"<caption>.*?</caption>", otsl_html)
            caption = caption_match.group(0).replace("<caption>", "").replace("</caption>", "").strip() if caption_match else ""
            caption = re.sub(r"<[^>]+>", "", caption).strip()

            # Extract headers (everything before first <nl>)
            header_part = re.search(r"(.*?)<nl>", otsl_html, re.DOTALL)
            headers = []
            if header_part:
                headers = re.findall(r"<ched>([^<]+)", header_part.group(1))

            # Extract rows
            rows = []
            row_parts = re.split(r"<nl>", otsl_html)
            if headers:
                rows_to_extract = row_parts[1:]
            else:
                rows_to_extract = row_parts
            for row_html in rows_to_extract:  # skip header row
                cells = re.findall(r"<fcel>([^<]+)", row_html)
                if cells:
                    rows.append(cells)

            # Build markdown table
            md = ""
            if headers:
                md += "| " + " | ".join(headers) + " |\n"
                md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for row in rows:
                md += "| " + " | ".join(row) + " |\n"

            if caption:
                md = f"**{caption}**\n\n" + md

            return md.strip()
        except Exception as e:
            logging.error(f"Error parsing otsl: {str(e)}")

    def parse_doctag(self):
        try:
            if self.doctagstr:
                start_time = time.time()
                # Clean doctag string
                doctagstr_pre = [s.strip()+"</doctag>" for s in self.doctagstr.split("</doctag>")][:-1]
                doctagstr_clean = []
                for docpage in doctagstr_pre:
                    docpage = self.clean_truncated_before_end(docpage)
                    docpage = self.remove_dangling_tags(docpage)
                    doctagstr_clean.append(docpage)
                
                doctagstr_clean = "\n\n".join(doctagstr_clean)
                self.soup = BeautifulSoup(doctagstr_clean, "html.parser")
                pages = self.soup.find_all("doctag")
                for p, page in enumerate(pages):                    
                    # Handle <picture> tags
                    pic_id = 0
                    for pic in page.find_all("picture"):
                        image_flag = self.parse_image(pic, p, pic_id)
                        if image_flag:
                            pic.replace_with("[Image]")
                            pic_id += 1
                        else:
                            pic.replace_with("")

                    # Handle <otsl> tables
                    for table in page.find_all("otsl"):
                        table_md = self.parse_otsl(table)
                        table.replace_with(table_md)

                    # Strip all <loc_###> tags but keep their content
                    for loc_tag in page.find_all():
                        if loc_tag.name.startswith("loc_"):
                            loc_tag.unwrap()

                    # Optional: Keep section headers bolded
                    for sh in self.soup.find_all(re.compile(r"section_header_level_\d+")):
                        text = sh.get_text(strip=True)
                        if not text.startswith("**") and not text.endswith("**"):
                            sh.string = f"**{text}**"
                        else:
                            sh.string = text
                end_time = time.time()
                processing_time = end_time - start_time
                logging.info(f"DocTags parsed successfully in {processing_time:.2f} seconds")
                return self.soup.get_text(separator="\n\n", strip=True)
        except Exception as e:
            logging.error(f"Error parsing doctag: {str(e)}")
