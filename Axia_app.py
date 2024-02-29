import streamlit as st
import pandas as pd
import pandas_gbq
import pandas 
import os
from google.oauth2 import service_account
from google.cloud import bigquery
from datetime import datetime, timedelta
from scipy.stats import chi2_contingency
from PIL import Image
from git import Repo
import base64
import requests
import json
from google.cloud import storage

credentials = service_account.Credentials.from_service_account_info(
          st.secrets["gcp_service_account"]
      )

Account = "Axia"
client = bigquery.Client(credentials=credentials)
bucket_name = "creativetesting_images_axia"
main_table_id = 'axia-414123.axia_segments.ad_level_data'
creativetesting_table_id = 'axia-414123.axia_streamlit.creativetestingstorage'
correct_hashed_password = "CFAxiaCreativeTest0947$"

st.set_page_config(page_title= f"{Account} Creative Ad Testing Dash",page_icon="🧑‍🚀",layout="wide")

def initialize_storage_client():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    storage_client = storage.Client(credentials=credentials)
    return storage_client

# Use this client for GCS operations
storage_client = initialize_storage_client()


def password_protection():
  if 'authenticated' not in st.session_state:
      st.session_state.authenticated = False
      
  if not st.session_state.authenticated:
      password = st.text_input("Enter Password:", type="password")
      
      if st.button("Login"):
          if password == correct_hashed_password:
              st.session_state.authenticated = True
              main_dashboard()
          else:
              st.error("Incorrect Password. Please try again or contact the administrator.")
  else:
      main_dashboard()


def download_blob_to_temp(bucket_name, source_blob_name, temp_folder="/tmp"):
    """Downloads a blob from the bucket to a temporary file."""
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Create a temporary file path
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    local_path = os.path.join(temp_folder, source_blob_name)

    # Download the blob to the temporary file path
    blob.download_to_filename(local_path)

    return local_path


def update_ad_set_table(test_name, ad_names):
    # Query to find the current Ad-Set and Campaign
    query = """
    SELECT Test_Name, Ad_Names FROM `axia-414123.axia_streamlit.creativetestingstorage` WHERE Type = 'Current'
    """
    current_ad_test = pandas.read_gbq(query, credentials=credentials)

    # If current Ad-Set exists, update it to 'Past'
    if not current_ad_test.empty:
        update_query = """
        UPDATE `axia-414123.axia_streamlit.creativetestingstorage`
        SET Type = 'Past'
        WHERE Test_Name = @current_ad_test 
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("current_ad_test", "STRING", current_ad_test.iloc[0]['Test_Name'])
            ]
        )
        client.query(update_query, job_config=job_config).result()

    # Insert the new Ad-Set with Type 'Current'
    insert_query = """
    INSERT INTO `axia-414123.axia_streamlit.creativetestingstorage` (Test_Name, Ad_Names, Type) VALUES (@new_ad_test, @ad_names, 'Current')
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("new_ad_test", "STRING", test_name),
            bigquery.ScalarQueryParameter("ad_names", "STRING", ad_names)
        ]
    )
    client.query(insert_query, job_config=job_config).result()
    st.success(f"Upload was successful! Please refresh the page to see updates.")


def upload_to_gcs(bucket_name, source_file, destination_blob_name):
    # Initialize the GCS client
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)

    # Create a new blob and upload the file's content.
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(source_file, content_type='image/jpeg')  # Set content_type as per your file type


### Code for past tests function ###
def process_ad_set_data(data, test, past_test_data):
    # Filter data for the specific ad set

    data = data.rename(columns={
      'Campaign_Name__Facebook_Ads': 'Campaign',
      'Ad_Set_Name__Facebook_Ads': 'Ad_Set',
      'Ad_Name__Facebook_Ads' : 'Ad_Name',
      'Impressions__Facebook_Ads' : 'Impressions',
      'Link_Clicks__Facebook_Ads' : 'Clicks',
      'Amount_Spent__Facebook_Ads' : 'Cost',
      'Leads__Facebook_Ads' : 'Leads',
      'Ad_Effective_Status__Facebook_Ads' : 'Ad_Status',
      'Ad_Preview_Shareable_Link__Facebook_Ads' : 'Ad_Link'
    })

    ad_names = past_test_data['Ad_Names'].iloc[0]
    ad_names = ad_names.split(",")

    # Filter data on just ad_set
    ad_set_data = data[data['Ad_Name'].isin(ad_names)]

    # Your data processing steps
    selected_columns = ['Ad_Name', 'Impressions', 'Clicks', 'Cost', 'Leads']
    filtered_data = ad_set_data[selected_columns]
    grouped_data = filtered_data.groupby(['Ad_Name']).sum()
    aggregated_data = grouped_data.reset_index()

    total = aggregated_data.sum(numeric_only=True)
    total['CPC'] = total['Cost']/total['Clicks']
    total['CPM'] = (total['Cost']/total['Impressions'])*1000
    total['CTR'] = total['Clicks']/total['Impressions']
    total['CVR'] = total['Leads']/total['Clicks']
    total['CPL'] = total['Cost']/total['Leads']
    total['Ad_Name'] = ""
    total['Ad_Set'] = 'Total'
  
    #Calculate cols
    aggregated_data['CPC'] = aggregated_data['Cost']/aggregated_data['Clicks']
    aggregated_data['CPM'] = (aggregated_data['Cost']/aggregated_data['Impressions'])*1000
    aggregated_data['CTR'] = aggregated_data['Clicks']/aggregated_data['Impressions']
    aggregated_data['CVR'] = aggregated_data['Leads']/aggregated_data['Clicks']
    aggregated_data['CPL'] = aggregated_data['Cost']/aggregated_data['Leads']

    #Sort leads so highest performer is at the top
    aggregated_data.sort_values(by='Leads', ascending=False, inplace=True)
  
    total_df = pd.DataFrame([total])
    # Reorder columns in total_df to match aggregated_data
    total_df = total_df[[ 'Ad_Name', 'Impressions', 'Clicks', 'Cost', 'Leads', 'CPL', 'CPC', 'CPM', 'CTR', 'CVR']]

    # Concatenate aggregated_data with total_df
    final_df = pd.concat([aggregated_data, total_df])

    # Initialize an empty list to store significance results
    significance_results = []
  
    # Top row data for comparison
    top_ad_leads = final_df.iloc[0]['Leads']
    top_ad_impressions = final_df.iloc[0]['Impressions']
  
    # Iterate through each row except the first and last
    for index, row in final_df.iloc[1:-1].iterrows():
        variant_leads = row['Leads']
        variant_impressions = row['Impressions']
  
        # Chi-square test
        chi2, p_value, _, _ = chi2_contingency([
            [top_ad_leads, top_ad_impressions - top_ad_leads],
            [variant_leads, variant_impressions - variant_leads]
        ])
  
        # Check if the result is significant and store the result
        significance_label = f"{p_value:.3f} - {'Significant' if p_value < 0.05 else 'Not significant'}"
        significance_results.append(significance_label)

    # Add a placeholder for the top row and append for the total row
    significance_results = [''] + significance_results + ['']
      
    # Add the significance results to the DataFrame
    final_df['Significance'] = significance_results

    column_order = ['Ad_Name', 'Cost', 'CPM', 'Clicks', 'CPC', 'CTR', 'Leads', 'CPL', 'CVR', 'Significance']
    final_df = final_df[column_order]
  
    final_df.reset_index(drop=True, inplace=True)

    #Format final_df correctly
    final_df['Cost'] = round(final_df['Cost'], 0).astype(int)
    final_df['Cost'] = final_df['Cost'].apply(lambda x: f"${x}")

    final_df['CPL'] = round(final_df['CPL'], 0).astype(int)
    #final_df['CPL'] = final_df['CPL'].apply(lambda x: f"${x}")
    final_df['CPL'] = final_df['CPL'].apply(lambda x: '' if abs(x) > 10000 else f"${x}")

    final_df['CPC'] = round(final_df['CPC'], 2)
    final_df['CPC'] = final_df['CPC'].apply(lambda x: f"${x}")

    final_df['CPM'] = round(final_df['CPM'], 0).astype(int)
    final_df['CPM'] = final_df['CPM'].apply(lambda x: f"${x}")

    final_df['CTR'] = final_df['CTR'].apply(lambda x: f"{x*100:.2f}%")
    final_df['CVR'] = final_df['CVR'].apply(lambda x: f"{x*100:.2f}%")   
          
    return final_df
    

# Function to create columns and display images with captions
def display_images(images, captions):
    num_images = len(images)
    cols = st.columns(num_images + 2)  # Extra columns for white space

    # Display images in the center columns
    for idx, image_name in enumerate(images):
        # Download the image from GCS to a temporary file
        image_name = image_name.replace("/", "-").replace("$","").replace(". "," ")
        local_image_path = download_blob_to_temp(bucket_name, image_name)

        with cols[idx + 1]:  # +1 for offset due to initial white space
            st.image(local_image_path, caption=captions[idx], use_column_width=True)


def main_dashboard():
  st.markdown(f"<h1 style='text-align: center;'>{Account} Creative Ad Testing</h1>", unsafe_allow_html=True)
  st.markdown("<h2 style='text-align: center;'>Current Test</h2>", unsafe_allow_html=True)
  # Calculate the date one year ago from today
  one_year_ago = (datetime.now() - timedelta(days=365)).date()
  
  if 'full_data' not in st.session_state:
      credentials = service_account.Credentials.from_service_account_info(
          st.secrets["gcp_service_account"]
      )
      client = bigquery.Client(credentials=credentials)
      # Modify the query
      query = f"""
      SELECT * FROM `{main_table_id}` 
      WHERE Date BETWEEN '{one_year_ago}' AND CURRENT_DATE() """
      
      st.session_state.full_data = pandas.read_gbq(query, credentials=credentials)

  data = st.session_state.full_data
  
  if 'current_test_data' not in st.session_state:
      credentials = service_account.Credentials.from_service_account_info(
          st.secrets["gcp_service_account"]
      )
      client = bigquery.Client(credentials=credentials)
      # Modify the query
      query = f"""
      SELECT * FROM `{creativetesting_table_id}` 
      WHERE Type = 'Current'"""
      st.session_state.current_test_data = pandas.read_gbq(query, credentials=credentials)

  current_test_data = st.session_state.current_test_data

  if 'past_test_data' not in st.session_state:
      credentials = service_account.Credentials.from_service_account_info(
          st.secrets["gcp_service_account"]
      )
      client = bigquery.Client(credentials=credentials)
      # Modify the query
      query = f"""
      SELECT * FROM `{creativetesting_table_id}` 
      WHERE Type = 'Past'"""
      st.session_state.past_test_data = pandas.read_gbq(query, credentials=credentials)

  past_test_data = st.session_state.past_test_data
  past_test_data['Test_Name'] = past_test_data['Test_Name'].apply(lambda x: x.strip("'"))
  past_test_data = past_test_data.iloc[::-1].reset_index(drop=True)
  
  # Renaming columns in a DataFrame
  data = data.rename(columns={
      'Campaign_Name__Facebook_Ads': 'Campaign',
      'Ad_Set_Name__Facebook_Ads': 'Ad_Set',
      'Ad_Name__Facebook_Ads' : 'Ad_Name',
      'Impressions__Facebook_Ads' : 'Impressions',
      'Link_Clicks__Facebook_Ads' : 'Clicks',
      'Amount_Spent__Facebook_Ads' : 'Cost',
      'Leads__Facebook_Ads' : 'Leads',
      'Ad_Effective_Status__Facebook_Ads' : 'Ad_Status',
      'Ad_Preview_Shareable_Link__Facebook_Ads' : 'Ad_Link'
  })


  # Streamlit interface for selecting new ad set
  with st.expander("Update Test and Upload Images"):
    test_name = st.text_input("Enter Test Name")
    number_of_ads = st.number_input("How many ad names do you want to enter?", min_value=1, format='%d')
    new_ad_names = []
    uploaded_images = {}
    all_filled = True 

    for i in range(int(number_of_ads)):
        ad_name = st.text_input(f"Ad Name {i+1}", key=f"ad_name_{i}")
              
        if ad_name:  # If there's an ad name entered
            ad_exists = data['Ad_Name'].str.contains(ad_name, regex=False).any()
            new_ad_names.append(ad_name)  # Store ad name
            if ad_exists:
                uploaded_file = st.file_uploader(f"Upload image for {ad_name}", key=f"uploaded_image_{i}", type=['png', 'jpg', 'jpeg'])
                uploaded_images[ad_name] = uploaded_file  # Associate uploaded file with ad name
                if uploaded_file is None:
                    all_filled = False  # Mark as not ready if any image is missing
        else:
            all_filled = False  # Mark as not ready if any ad name is missing

     # Enable the upload button only if all conditions are met
    if all_filled and st.button("Upload Images"):
        # Proceed with the upload logic
        for ad_name, uploaded_file in uploaded_images.items():
            if uploaded_file is not None:
                # Example: Upload logic here
                ad_name = ad_name.replace("/", "-").replace("$","").replace(". ", " ")
                upload_to_gcs(bucket_name, uploaded_file, f"{ad_name}.jpg")
                pass
        # Update the database with the new test name and associated ad names
        combined_ad_names = ",".join(new_ad_names)
        update_ad_set_table(test_name, combined_ad_names)
        st.success("Images uploaded and test updated successfully!")


  if current_test_data.empty:
            st.markdown("<h4 style='text-align: center;'>No Current Tests to Display</h4>", unsafe_allow_html=True)
  else:              
            current_Ad_Set = current_test_data['Test_Name'].iloc[0]

            # Get list of ad_names from ad names string 
            ad_names = current_test_data['Ad_Names'].iloc[0]
            ad_names = ad_names.split(',')
          
            current_Ad_Set = current_Ad_Set.strip("'")
          
            # Filter data on just ad_set
            
            ad_set_data = data[data['Ad_Name'].isin(ad_names)]
            
            data = ad_set_data
                    
            selected_columns = ['Ad_Name', 'Impressions', 'Clicks','Cost', 'Leads']
            
            filtered_data = data[selected_columns]
          
            # Grouping the data by 'Ad_Set'
            grouped_data = filtered_data.groupby(['Ad_Name'])
            
            # Summing up the numeric columns for each group
            aggregated_data = grouped_data.sum()
            
            # Reset the index
            aggregated_data.reset_index(inplace=True)
          
            total = aggregated_data.sum(numeric_only=True)
            total['CPC'] = total['Cost']/total['Clicks']
            total['CPM'] = (total['Cost']/total['Impressions'])*1000
            total['CTR'] = total['Clicks']/total['Impressions']
            total['CVR'] = total['Leads']/total['Clicks']
            total['CPL'] = total['Cost']/total['Leads']
            total['Ad_Name'] = ""
            total['Ad_Set'] = 'Total'
            
            #Calculate cols
            aggregated_data['CPC'] = aggregated_data['Cost']/aggregated_data['Clicks']
            aggregated_data['CPM'] = (aggregated_data['Cost']/aggregated_data['Impressions'])*1000
            aggregated_data['CTR'] = aggregated_data['Clicks']/aggregated_data['Impressions']
            aggregated_data['CVR'] = aggregated_data['Leads']/aggregated_data['Clicks']
            aggregated_data['CPL'] = aggregated_data['Cost']/aggregated_data['Leads']
          
            #Sort leads so highest performer is at the top
            aggregated_data.sort_values(by='Leads', ascending=False, inplace=True)
            
            total_df = pd.DataFrame([total])
            # Reorder columns in total_df to match aggregated_data
            total_df = total_df[['Ad_Set', 'Ad_Name', 'Impressions', 'Clicks', 'Cost', 'Leads', 'CPL', 'CPC', 'CPM', 'CTR', 'CVR']]
          
            # Concatenate aggregated_data with total_df
            final_df = pd.concat([aggregated_data, total_df])
          
            # Initialize an empty list to store significance results
            significance_results = []
            
            # Top row data for comparison
            top_ad_leads = final_df.iloc[0]['Leads']
            top_ad_impressions = final_df.iloc[0]['Impressions']
            
            # Iterate through each row except the first and last
            for index, row in final_df.iloc[1:-1].iterrows():
                variant_leads = row['Leads']
                variant_impressions = row['Impressions']
            
                # Chi-square test
                chi2, p_value, _, _ = chi2_contingency([
                    [top_ad_leads, top_ad_impressions - top_ad_leads],
                    [variant_leads, variant_impressions - variant_leads]
                ])
            
                # Check if the result is significant and store the result
                significance_label = f"{p_value:.3f} - {'Significant' if p_value < 0.05 else 'Not significant'}"
                significance_results.append(significance_label)
          
            # Add a placeholder for the top row and append for the total row
            significance_results = [''] + significance_results + ['']
            
            # Add the significance results to the DataFrame
            final_df['Significance'] = significance_results
          
            column_order = ['Ad_Name', 'Cost', 'CPM', 'Clicks', 'CPC', 'CTR', 'Leads', 'CPL', 'CVR', 'Significance']
            final_df = final_df[column_order]
          
            final_df.reset_index(drop=True, inplace=True)
          
            uploaded_images = []
            image_captions = []
          
          
            #Format final_df correctly
            final_df['Cost'] = round(final_df['Cost'], 0).astype(int)
            final_df['Cost'] = final_df['Cost'].apply(lambda x: f"${x}")
          
            final_df['CPL'] = round(final_df['CPL'], 0).astype(int)
            #final_df['CPL'] = final_df['CPL'].apply(lambda x: f"${x}")
            final_df['CPL'] = final_df['CPL'].apply(lambda x: '' if abs(x) > 10000 else f"${x}")
          
            final_df['CPC'] = round(final_df['CPC'], 2)
            final_df['CPC'] = final_df['CPC'].apply(lambda x: f"${x}")
          
            final_df['CPM'] = round(final_df['CPM'], 0).astype(int)
            final_df['CPM'] = final_df['CPM'].apply(lambda x: f"${x}")
          
            final_df['CTR'] = final_df['CTR'].apply(lambda x: f"{x*100:.2f}%")
            final_df['CVR'] = final_df['CVR'].apply(lambda x: f"{x*100:.2f}%")
                    
            # Display the aggregated data
            st.dataframe(final_df, width=2000)

            #Get list of ad_names for images
            final_adnames = final_df['Ad_Name']
            final_adnames = [item + ".jpg" for item in final_adnames]
            final_adnames.pop()

            display_images(final_adnames, final_adnames)        
          
  st.markdown("<h2 style='text-align: center;'>Past Tests</h2>", unsafe_allow_html=True)

  if past_test_data.empty:
            st.markdown("<h4 style='text-align: center;'>No Past Tests to Display</h4>", unsafe_allow_html=True)
  else:        
            past_tests = past_test_data['Test_Name']
          
            # Dictionary to store DataFrames for each ad set
            test_dfs = {}
            
            for test in past_tests:
                test_dfs[test] = process_ad_set_data(st.session_state.full_data, test, past_test_data)
          
            for test in test_dfs:
                with st.expander(f"Show Data for {test}"):
                    st.dataframe(test_dfs[test], width=2000)
                    current_df = test_dfs[test]
                    ad_names = current_df['Ad_Name']
                    ad_names = [item + ".jpg" for item in ad_names]
                    ad_names.pop()
                    display_images(ad_names, ad_names)

if __name__ == '__main__':
    password_protection()

    
