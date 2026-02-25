"""
Module: parse_documents.py
Functionality: Hybrid PDF parsing using both PyMuPDF and pdfplumber for optimal table and text extraction.
Combines the strengths of both libraries while maintaining document flow order.
"""
import os
import re
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import fitz  # PyMuPDF for general text and positioning

# Try to import pdfplumber for enhanced table detection
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️ pdfplumber not available. Install with: pip install pdfplumber for enhanced table detection")

# Try to import pandas for legacy compatibility
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

def _is_valid_table(cleaned_data: List[List[str]]) -> bool:
    """
    Validate if detected table is actually a table structure or just formatted text.
    
    Args:
        cleaned_data: List of rows with cells
        
    Returns:
        True if it's a valid table, False otherwise
    """
    if len(cleaned_data) < 3:  # Need at least 3 rows (header + 2 data rows)
        return False
    
    # Check if rows have consistent column structure
    row_lengths = [len(row) for row in cleaned_data]
    if len(set(row_lengths)) > 2:  # Too much variation in column count
        return False
    
    # Check if it has proper tabular data (not just a paragraph split into lines)
    max_cols = max(row_lengths)
    if max_cols < 2:  # Single column is likely just text
        return False
    
    # Check for table-like characteristics
    # 1. Multiple rows should have multiple non-empty cells
    multi_cell_rows = 0
    for row in cleaned_data:
        non_empty_cells = sum(1 for cell in row if cell and cell.strip())
        if non_empty_cells >= 2:
            multi_cell_rows += 1
    
    # If less than 50% of rows have multiple cells, it's probably not a table
    if multi_cell_rows / len(cleaned_data) < 0.5:
        return False
    
    # 2. Check for list-like patterns (single long text in first column, others empty)
    list_pattern_count = 0
    for row in cleaned_data:
        if len(row) >= 2:
            first_cell = row[0].strip() if row[0] else ""
            other_cells = [cell.strip() for cell in row[1:] if cell]
            
            # If first cell is very long and others are mostly empty, it's likely a list
            if len(first_cell) > 50 and len(other_cells) <= 1:
                list_pattern_count += 1
    
    # If more than 60% rows follow list pattern, reject as table
    if list_pattern_count / len(cleaned_data) > 0.6:
        return False
    
    # 3. Check for paragraph-like content (very long text cells)
    long_text_cells = 0
    total_cells = 0
    for row in cleaned_data:
        for cell in row:
            if cell and cell.strip():
                total_cells += 1
                if len(cell.strip()) > 100:  # Very long text
                    long_text_cells += 1
    
    # If too many cells have very long text, it's probably paragraphs
    if total_cells > 0 and long_text_cells / total_cells > 0.7:
        return False
    
    return True

def _is_headerless(table_data: List[List[str]]) -> bool:
    """
    Check if a table is likely headerless based on the first row's content.
    A row is considered a data row if it contains numbers or long text.
    """
    if not table_data:
        return True
    
    first_row = table_data[0]
    # Simple heuristic: if any cell in the first row contains only numbers, it's likely not a header.
    for cell in first_row:
        cell_text = cell.strip()
        if cell_text.isdigit():
            return True
    
    # Another heuristic: if the first row is very similar to the second row
    if len(table_data) > 1:
        second_row = table_data[1]
        if len(first_row) == len(second_row):
            # Check for structural similarity (e.g., both have numbers in the same places)
            first_row_types = ['d' if c.isdigit() else 's' for c in first_row]
            second_row_types = ['d' if c.isdigit() else 's' for c in second_row]
            if first_row_types == second_row_types:
                return True

    return False

def _find_contextual_header(table_bbox: List[float], text_blocks: List[Dict], page_height: float) -> str:
    """
    Find the most likely header text for a table from nearby text blocks.
    Prioritizes text that is close, directly above, and centrally aligned.
    """
    best_candidate = ""
    min_distance = float('inf')

    table_y_top = table_bbox[1]

    for block in text_blocks:
        block_bbox = block.get('bbox', [0,0,0,0])
        block_y_bottom = block_bbox[3]
        
        # Rule 1: Text must be above the table
        if block_y_bottom <= table_y_top:
            vertical_distance = table_y_top - block_y_bottom
            
            # Rule 2: Text must be reasonably close vertically (e.g., within 10% of page height)
            if vertical_distance < page_height * 0.1:

                    # This is a good candidate, check if it's the best one so far
                if vertical_distance < min_distance:
                    min_distance = vertical_distance
                    best_candidate = block.get('content', '')

    return best_candidate.strip()


def extract_tables_with_pdfplumber(pdf_path: str) -> Dict[int, List[Dict]]:
    """
    Extract tables using pdfplumber for superior table detection.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary mapping page numbers to list of tables with positioning
    """
    page_tables = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = []
                
                # Extract tables with pdfplumber's superior detection - IMPROVED SETTINGS
                detected_tables = page.find_tables(table_settings={
                    "vertical_strategy": "lines", 
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 10,  # Increased from 3 to 10 for better line detection
                    "min_words_vertical": 4,  # Increased from 2 to 4 
                    "min_words_horizontal": 2  # Increased from 1 to 2
                })
                
                # We need all text blocks on the page to find contextual headers
                # This is done once per page for efficiency
                all_page_text_blocks = [{'content': w['text'], 'bbox': (w['x0'], w['top'], w['x1'], w['bottom'])} for w in page.extract_words(keep_blank_chars=False, use_text_flow=True)]

                for table in detected_tables:
                    try:
                        # Extract table data
                        table_data = table.extract()
                        
                        if table_data and len(table_data) >= 2:  # At least header + 1 row
                            # Clean and process table data
                            cleaned_data = []
                            for row in table_data:
                                if row and any(cell and str(cell).strip() for cell in row if cell is not None):
                                    cleaned_row = []
                                    for cell in row:
                                        cell_str = str(cell).strip() if cell is not None else ""
                                        # Clean common PDF artifacts
                                        cell_str = re.sub(r'\s+', ' ', cell_str)
                                        cell_str = re.sub(r'[^\x20-\x7E\u00A0-\u024F\u1E00-\u1EFF]', '', cell_str)
                                        cleaned_row.append(cell_str)
                                    if any(cell for cell in cleaned_row):  # Only add non-empty rows
                                        cleaned_data.append(cleaned_row)
                            
                            if len(cleaned_data) >= 2:  # Ensure we have meaningful data
                                # VALIDATE IF THIS IS ACTUALLY A TABLE
                                if not _is_valid_table(cleaned_data):
                                    continue  # Skip this "table" as it's likely just formatted text
                                
                                # Normalize column count
                                max_cols = max(len(row) for row in cleaned_data)
                                normalized_data = []
                                for row in cleaned_data:
                                    while len(row) < max_cols:
                                        row.append("")
                                    normalized_data.append(row[:max_cols])
                                
                                # Get table bbox for positioning
                                bbox = table.bbox if hasattr(table, 'bbox') else [0, 0, 0, 0]
                                
                                # Check if table is headerless
                                is_headerless = _is_headerless(normalized_data)
                                table_data_for_markdown = list(normalized_data) # Make a copy

                                if is_headerless:
                                    # Find header from nearby text using all text blocks from the page
                                    summary = _find_contextual_header(bbox, all_page_text_blocks, page.height)
                                    if not summary:
                                        summary = f"This is a table with {max_cols} columns and {len(table_data_for_markdown)} rows."
                                    
                                    # Add generic headers for markdown creation
                                    generic_headers = [f"Column {i+1}" for i in range(max_cols)]
                                    table_data_for_markdown.insert(0, generic_headers)
                                else:
                                    # Generate a descriptive summary for the table with real headers
                                    rows = len(normalized_data) - 1 # Exclude header
                                    cols = len(normalized_data[0]) if normalized_data else 0
                                    col_names = ", ".join(f"'{c}'" for c in normalized_data[0]) if normalized_data else ""
                                    summary = f"This is a table with {cols} columns and {rows} rows. The columns are: {col_names}."

                                # Create markdown table using the potentially modified data
                                table_markdown = _create_clean_markdown_table(table_data_for_markdown)
                                
                                # Create the final hybrid content with summary + markdown table
                                hybrid_content = f"{summary}\n\n{table_markdown}"

                                tables.append({
                                    'content': hybrid_content,
                                    'type': 'table',
                                    'bbox': bbox,
                                    'y_position': bbox[1] if bbox else 0,
                                    'rows': len(normalized_data),
                                    'cols': max_cols,
                                    'raw_data': normalized_data
                                })
                                
                    except Exception as e:
                        print(f"⚠️ Table processing error on page {page_num + 1}: {e}")
                        continue
                
                if tables:
                    page_tables[page_num] = tables
                    
    except Exception as e:
        print(f"⚠️ pdfplumber table extraction error: {e}")
    
    return page_tables

def _create_clean_markdown_table(data: List[List[str]]) -> str:
    """Create a clean, well-formatted markdown table without truncating content."""
    if not data or len(data) < 2:
        return ""
    
    # Use first row as headers, rest as data
    headers = data[0]
    rows = data[1:]
    
    # Clean headers and ensure they have meaningful names
    clean_headers = []
    for i, header in enumerate(headers):
        header_str = str(header).strip() if header else ""
        if not header_str:
            header_str = f"Col_{i+1}"
        clean_headers.append(header_str)
    
    # Create formatted table WITHOUT width restrictions
    table_lines = []
    
    # Header row
    header_cells = [str(header).strip() for header in clean_headers]
    table_lines.append("| " + " | ".join(header_cells) + " |")
    
    # Separator row  
    separator_cells = ["-" * max(3, len(str(header))) for header in clean_headers]
    table_lines.append("| " + " | ".join(separator_cells) + " |")
    
    # Data rows - NO TRUNCATION, preserve all content
    for row in rows:
        data_cells = []
        for cell in row:
            cell_str = str(cell).strip() if cell else ""
            # Clean whitespace only, preserve all content
            cell_str = re.sub(r'\s+', ' ', cell_str)
            data_cells.append(cell_str)
        table_lines.append("| " + " | ".join(data_cells) + " |")
    
    return "\n".join(table_lines)

def extract_text_with_pymupdf(pdf_path: str, page_tables: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
    """
    Extract text blocks using PyMuPDF while avoiding table content areas.
    
    Args:
        pdf_path: Path to the PDF file
        page_tables: Dictionary of tables per page for overlap detection
        
    Returns:
        Dictionary mapping page numbers to list of text blocks
    """
    page_text_blocks = {}
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_blocks = []
            
            # Get tables for this page to avoid overlaps
            tables_on_page = page_tables.get(page_num, [])
            table_bboxes = [table['bbox'] for table in tables_on_page]
            
            # Extract text blocks with positioning
            blocks = page.get_text("dict")
            
            for block_num, block in enumerate(blocks.get("blocks", [])):
                if "lines" in block:  # Text block
                    block_text = ""
                    font_sizes = []
                    is_bold = False
                    
                    for line in block["lines"]:
                        line_text = ""
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                line_text += text + " "
                                
                                # Collect font information
                                font_size = span.get("size", 12)
                                font_flags = span.get("flags", 0)
                                font_sizes.append(font_size)
                                
                                # Check if bold (bit 4 of flags)
                                if font_flags & (1 << 4):
                                    is_bold = True
                        
                        if line_text.strip():
                            block_text += line_text.strip() + " "
                    
                    if block_text.strip():
                        bbox = block.get("bbox", [0, 0, 0, 0])
                        
                        # Check if this text block overlaps with any table
                        overlaps_with_table = any(
                            _bbox_overlap(bbox, table_bbox, 0.5) 
                            for table_bbox in table_bboxes
                        )
                        
                        if not overlaps_with_table:
                            avg_font_size = np.mean(font_sizes) if font_sizes else 12
                            
                            # Determine content type based on formatting
                            content_type = "text"
                            if is_bold and avg_font_size > 14:
                                content_type = "heading"
                            elif is_bold:
                                content_type = "subheading"
                            
                            text_blocks.append({
                                'content': block_text.strip(),
                                'type': content_type,
                                'bbox': bbox,
                                'y_position': bbox[1],
                                'font_size': avg_font_size,
                                'is_bold': is_bold,
                                'block_number': block_num
                            })
            
            if text_blocks:
                page_text_blocks[page_num] = text_blocks
        
        doc.close()
        
    except Exception as e:
        print(f"⚠️ PyMuPDF text extraction error: {e}")
    
    return page_text_blocks

def _bbox_overlap(bbox1: List[float], bbox2: List[float], overlap_threshold: float = 0.5) -> bool:
    """Check if two bounding boxes overlap significantly."""
    if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
        return False
    
    # Calculate intersection
    x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
    y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
    
    if x_overlap <= 0 or y_overlap <= 0:
        return False
    
    intersection_area = x_overlap * y_overlap
    
    # Calculate areas
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    if area1 <= 0 or area2 <= 0:
        return False
    
    # Check if intersection is significant relative to the smaller box
    smaller_area = min(area1, area2)
    overlap_ratio = intersection_area / smaller_area
    
    return overlap_ratio >= overlap_threshold

def merge_page_content(page_num: int, text_blocks: List[Dict], tables: List[Dict]) -> List[Dict]:
    """
    Merge text blocks and tables for a page by their position to maintain document order.
    
    Args:
        page_num: Page number
        text_blocks: List of text blocks for the page
        tables: List of tables for the page
        
    Returns:
        List of content elements in document order
    """
    all_content = []
    
    # Add text blocks
    for block in text_blocks:
        all_content.append({
            'content': block['content'],
            'type': block['type'],
            'y_position': block['y_position'],
            'page': page_num + 1,
            'bbox': block['bbox'],
            'font_size': block.get('font_size', 12),
            'is_bold': block.get('is_bold', False)
        })
    
    # Add tables
    for table in tables:
        all_content.append({
            'content': table['content'],
            'type': 'table',
            'y_position': table['y_position'],
            'page': page_num + 1,
            'bbox': table['bbox'],
            'rows': table['rows'],
            'cols': table['cols']
        })
    
    # Sort by vertical position (top to bottom)
    all_content.sort(key=lambda x: x['y_position'])
    
    return all_content

def clean_text_content(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove excessive whitespace while preserving structure
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks to double
    
    # Clean common PDF artifacts
    text = re.sub(r'[^\x20-\x7E\u00A0-\u024F\u1E00-\u1EFF\n\r\t]', '', text)
    
    # Fix common character issues
    text = text.replace('�', '')  # Remove replacement characters
    
    return text.strip()

def parse_document_hybrid(pdf_path: str, save_parsed_text: bool = False) -> dict:
    """
    Hybrid PDF parsing using both PyMuPDF and pdfplumber for optimal results.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing parsed content with tables and text in document order
    """
    try:
        doc_name = os.path.basename(pdf_path)
        print(f"🚀 Starting hybrid parsing for {doc_name}...")
        print(f"📊 Using pdfplumber for tables + PyMuPDF for text")
        
        start_time = time.time()
        
        # Step 1: Extract tables using pdfplumber (superior table detection)
        print("🔍 Extracting tables with pdfplumber...")
        page_tables = extract_tables_with_pdfplumber(pdf_path)
        total_tables = sum(len(tables) for tables in page_tables.values())
        print(f"✅ Found {total_tables} tables across {len(page_tables)} pages")
        
        # Step 2: Extract text using PyMuPDF (avoiding table areas)
        print("📝 Extracting text with PyMuPDF...")
        page_text_blocks = extract_text_with_pymupdf(pdf_path, page_tables)
        total_text_blocks = sum(len(blocks) for blocks in page_text_blocks.values())
        print(f"✅ Found {total_text_blocks} text blocks")
        
        # Step 3: Merge content by page and position
        print("🔄 Merging content in document order...")
        all_content = []
        
        # Get total page count
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
        
        for page_num in range(total_pages):
            text_blocks = page_text_blocks.get(page_num, [])
            tables = page_tables.get(page_num, [])
            
            page_content = merge_page_content(page_num, text_blocks, tables)
            all_content.extend(page_content)
        
        # Step 4: Generate final output
        final_text = ""
        for item in all_content:
            content = item['content'].strip()
            
            if item['type'] == 'table':
                # Format tables with clear separation
                final_text += "\n" + "="*60 + "\n"
                final_text += "TABLE:\n"
                final_text += "="*60 + "\n\n"
                final_text += content + "\n\n"
                final_text += "="*60 + "\n\n"
            elif item['type'] == 'heading':
                # Format headings with separation
                final_text += "\n" + content + "\n"
                final_text += "-" * min(len(content), 60) + "\n\n"
            else:
                # Add regular text content
                cleaned_content = clean_text_content(content)
                if cleaned_content:
                    final_text += cleaned_content + "\n\n"
        
        processing_time = time.time() - start_time
        
        result = {
            'document_name': doc_name,
            'content': final_text.strip(),
            'total_pages': total_pages,
            'parsing_method': 'hybrid_pymupdf_pdfplumber',
            'processing_time': processing_time,
            'metadata': {
                'total_elements': len(all_content),
                'text_elements': total_text_blocks,
                'table_elements': total_tables,
                'pages_processed': total_pages,
                'characters_extracted': len(final_text)
            }
        }
        
        print(f"✅ Hybrid parsing complete!")
        print(f"📊 Results: {total_tables} tables, {total_text_blocks} text blocks")
        print(f"📄 Total: {len(final_text):,} characters in {processing_time:.2f}s")
        
        # Save parsed content if requested
        if save_parsed_text:
            save_parsed_content_to_file(result, pdf_path)
        
        return result
        
    except Exception as e:
        print(f"❌ Hybrid parsing error: {e}")
        return {
            'document_name': os.path.basename(pdf_path),
            'content': "",
            'total_pages': 0,
            'parsing_method': 'hybrid_error',
            'processing_time': 0,
            'metadata': {
                'total_elements': 0,
                'text_elements': 0,
                'table_elements': 0,
                'pages_processed': 0,
                'characters_extracted': 0,
                'error': str(e)
            }
        }

def save_parsed_content_to_file(result: dict, original_pdf_path: str) -> str:
    """
    Save the hybrid parsing result to a single clean output file.
    
    Args:
        result: Parsing result dictionary
        original_pdf_path: Path to original PDF file
        
    Returns:
        Path to the saved output file
    """
    try:
        # Create output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        base_name = os.path.splitext(os.path.basename(original_pdf_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_parsed.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"HYBRID PDF PARSING OUTPUT\n")
            f.write(f"="*50 + "\n")
            f.write(f"Document: {result['document_name']}\n")
            f.write(f"Method: {result['parsing_method']}\n")
            f.write(f"Pages: {result['total_pages']}\n")
            f.write(f"Tables: {result['metadata']['table_elements']}\n")
            f.write(f"Text Blocks: {result['metadata']['text_elements']}\n")
            f.write(f"Processing Time: {result['processing_time']:.2f}s\n")
            f.write(f"="*50 + "\n\n")
            
            # Write content
            f.write(result['content'])
        
        print(f"💾 Output saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"⚠️ Error saving hybrid output: {e}")
        return ""

# Legacy function names for backward compatibility
def parse_document_enhanced_tables(pdf_path: str, save_parsed_text: bool = False) -> dict:
    """Legacy function name - redirects to hybrid parsing."""
    return parse_document_hybrid(pdf_path, save_parsed_text)

def parse_document_with_tables(pdf_path: str, save_parsed_text: bool = False) -> dict:
    """Legacy function name - redirects to hybrid parsing."""
    return parse_document_hybrid(pdf_path, save_parsed_text)

def detect_table_structures(page, min_rows=2, min_cols=2) -> List[Dict]:
    """Legacy function for backward compatibility - uses PyMuPDF fallback method."""
    tables = []
    
    try:
        # Use PyMuPDF's find_tables method with enhanced settings
        if hasattr(page, 'find_tables'):
            found_tables = page.find_tables(
                vertical_strategy="lines_strict",
                horizontal_strategy="lines_strict",
                snap_tolerance=3.0,
                join_tolerance=3.0,
                edge_min_length=3.0,
                min_words_vertical=3,
                min_words_horizontal=1
            )
            
            for table in found_tables:
                try:
                    # Extract table data
                    table_data = table.extract()
                    bbox = table.bbox if hasattr(table, 'bbox') else [0, 0, 0, 0]
                    
                    if table_data and len(table_data) >= min_rows:
                        # Filter out empty rows and columns
                        filtered_data = []
                        for row in table_data:
                            if row and any(cell and str(cell).strip() for cell in row):
                                cleaned_row = []
                                for cell in row:
                                    cell_str = str(cell).strip() if cell else ""
                                    # Clean common PDF artifacts
                                    cell_str = re.sub(r'\s+', ' ', cell_str)
                                    cell_str = re.sub(r'[^\x20-\x7E\u00A0-\u024F\u1E00-\u1EFF]', '', cell_str)
                                    cleaned_row.append(cell_str)
                                filtered_data.append(cleaned_row)
                        
                        if len(filtered_data) >= min_rows and len(filtered_data[0]) >= min_cols:
                            # Create table with proper headers
                            try:
                                # Ensure all rows have the same number of columns
                                max_cols = max(len(row) for row in filtered_data)
                                normalized_data = []
                                for row in filtered_data:
                                    while len(row) < max_cols:
                                        row.append("")
                                    normalized_data.append(row[:max_cols])
                                
                                if len(normalized_data) >= min_rows and max_cols >= min_cols:
                                    # Create markdown table
                                    table_markdown = _create_clean_markdown_table(normalized_data)
                                    
                                    tables.append({
                                        'content': table_markdown,
                                        'type': 'table',
                                        'rows': len(normalized_data),
                                        'cols': max_cols,
                                        'bbox': bbox,
                                        'y_position': bbox[1] if bbox else 0,
                                        'raw_data': normalized_data
                                    })
                                    
                            except Exception as e:
                                print(f"⚠️ Table processing error: {e}")
                                continue
                                
                except Exception as e:
                    print(f"⚠️ Table extraction error: {e}")
                    continue
        
    except Exception as e:
        print(f"⚠️ Table detection error: {e}")
    
    return tables

if __name__ == "__main__":
    # Test the hybrid parser
    pdf_path = "docs/policy.pdf"
    if os.path.exists(pdf_path):
        result = parse_document_hybrid(pdf_path, save_parsed_text=True)
        print(f"✅ Parsing complete: {result['metadata']['characters_extracted']:,} characters")
    else:
        print(f"❌ PDF file not found: {pdf_path}")