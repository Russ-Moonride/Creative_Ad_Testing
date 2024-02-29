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
client = bigquery.Client(credentials=credentials)

#### Information to be changed when switching accounts ###
Account = "Ello"
bucket_name = "creativetesting_images_ello" 
main_table_id = "ello-407319.ello_Segments.ello_ad_level"
creativetesting_table_id = "ello-407319.ello_streamlit.CreativeTestingStorage"
correct_hashed_password = "CFElloAdTest2059$"


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



def filter_ad_names_by_campaign(ad_set, campaign_name, full_data):

    #Fiter to the ad_set
    filtered_data = full_data[full_data['Ad_Set_Name__Facebook_Ads'] == ad_set] 
          
    # Filter the full_data DataFrame for the given campaign name   
    filtered_data = full_data[full_data['Campaign_Name__Facebook_Ads'] == campaign_name]

    # Filter ad_names based on those present in the filtered_data
    filtered_ad_names = filtered_data["Ad_Name__Facebook_Ads"].unique()

    return filtered_ad_names

def get_campaign_value(ad_set, creative_storage_data):
    # Filter the creative_storage_data for the given ad_set
    filtered_data = creative_storage_data[creative_storage_data['Ad_Set'] == ad_set]

    # Check if there is an associated campaign value
    if not filtered_data.empty and 'Campaign' in filtered_data.columns:
        # Return the first campaign value found
        return filtered_data.iloc[0]['Campaign']
    else:
        # Return None if no campaign value is found
        return None

def update_ad_set_table(new_ad_set_name, campaign_name=None):
    # Query to find the current Ad-Set and Campaign
    query = """
    SELECT Ad_Set, Campaign FROM `{creativetesting_table_id}` WHERE Type = 'Current'
    """
    current_ad_set_campaign = pandas.read_gbq(query, credentials=credentials)

    # If current Ad-Set exists, update it to 'Past'
    if not current_ad_set_campaign.empty:
        update_query = """
        UPDATE `{creativetesting_table_id}`
        SET Type = 'Past'
        WHERE Ad_Set = @current_ad_set 
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("current_ad_set", "STRING", current_ad_set_campaign.iloc[0]['Ad_Set']),
                bigquery.ScalarQueryParameter("current_campaign", "STRING", current_ad_set_campaign.iloc[0]['Campaign'])
            ]
        )
        client.query(update_query, job_config=job_config).result()

    # Insert the new Ad-Set with Type 'Current'
    insert_query = """
    INSERT INTO `{creativetesting_table_id}` (Ad_Set, Campaign, Type) VALUES (@new_ad_set, @campaign, 'Current')
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("new_ad_set", "STRING", new_ad_set_name),
            bigquery.ScalarQueryParameter("campaign", "STRING", campaign_name if campaign_name else '')
        ]
    )
    client.query(insert_query, job_config=job_config).result()
    st.success(f"Upload was successful! Please refresh the page to see updates.")


def update_ad_set_if_exists(new_ad_set_name, uploaded_images, full_data, bucket_name, campaign_name=None):
    # The logic for retrieving and filtering ad_names is now handled in the main function
    # So, we directly proceed with uploading files and updating the ad set table

    for ad_name, uploaded_file in uploaded_images.items():
        destination_blob_name = f"{ad_name}.jpg" 
        upload_to_gcs(bucket_name, uploaded_file, destination_blob_name)
    
    update_ad_set_table(new_ad_set_name, campaign_name)  # Update the ad set table after successful uploads



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


def delete_ad_set(ad_set_value_to_delete, full_data):
        # SQL statement for deletion
        if ad_set_value_to_delete in full_data['Ad_Set_Name__Facebook_Ads'].values:
                  delete_query = """
                  DELETE FROM `{creativetesting_table_id}`
                  WHERE Ad_Set = @ad_set_value
                  AND Type = 'Past'
                  """
                  # Configure query parameters
                  job_config = bigquery.QueryJobConfig(
                      query_parameters=[
                          bigquery.ScalarQueryParameter("ad_set_value", "STRING", ad_set_value_to_delete)
                      ]
                  )
                  # Execute the query
                  client.query(delete_query, job_config=job_config).result()
                  st.experimental_rerun()
        else:
                  st.error("Ad_Set does not exist")


### Code for past tests function ###
def process_ad_set_data(data, ad_set, past_test_data):
    # Filter data for the specific ad set

    data = data.rename(columns={
      'Campaign_Name__Facebook_Ads': 'Campaign',
      'Ad_Set_Name__Facebook_Ads': 'Ad_Set',
      'Ad_Name__Facebook_Ads' : 'Ad_Name',
      'Impressions__Facebook_Ads' : 'Impressions',
      'Link_Clicks__Facebook_Ads' : 'Clicks',
      'Amount_Spent__Facebook_Ads' : 'Cost',
      'Lead_Submit_SunPower__Facebook_Ads' : 'Leads',
      'Ad_Effective_Status__Facebook_Ads' : 'Ad_Status',
      'Ad_Preview_Shareable_Link__Facebook_Ads' : 'Ad_Link'
    })

    campaign_value = get_campaign_value(ad_set, past_test_data)

    if campaign_value:
        # Filter data on both ad_set and campaign_value
        ad_set_data = data[(data['Ad_Set'] == ad_set) & (data['Campaign'] == campaign_value)]
    else:
        # Filter data on just ad_set
        ad_set_data = data[data['Ad_Set'] == ad_set]

    
    #ad_set_data = data[data['Ad_Set'] == ad_set]

    # Your data processing steps
    selected_columns = ['Ad_Set', 'Ad_Name', 'Impressions', 'Clicks', 'Cost', 'Leads']
    filtered_data = ad_set_data[selected_columns]
    grouped_data = filtered_data.groupby(['Ad_Set', 'Ad_Name']).sum()
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

    column_order = ['Ad_Set', 'Ad_Name', 'Cost', 'CPM', 'Clicks', 'CPC', 'CTR', 'Leads', 'CPL', 'CVR', 'Significance']
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


    final_df = final_df[final_df['Ad_Name'] != "specimen-v13_image_paid-social-LP_PID"]
    final_df = final_df[final_df['Ad_Name'] != "specimen-v7_image_paid-social-LP_PID"]      
          
    return final_df


def update_current_tests(new_ad_set_name, uploaded_files, full_data, bucket_name):
    ad_names = get_ad_names(new_ad_set_name, full_data)
    
    if len(uploaded_files) != len(ad_names):
        st.error(f"Please upload exactly {len(ad_names)} images for the ad names in this set.")
        return
    
    # Upload each file to GCS and update the ad set table
    for i, file in enumerate(uploaded_files):
        destination_blob_name = f"{new_ad_set_name}/{ad_names[i]}.jpg"  # Customize as needed
        upload_to_gcs(bucket_name, file, destination_blob_name)
    
    update_ad_set_table(new_ad_set_name)  # Update the ad set table after successful uploads


def get_ad_names(ad_set_name, ad_data):
    # Retrieve all ad names from the given ad set
    ad_names = ad_data[ad_data['Ad_Set_Name__Facebook_Ads'] == ad_set_name]['Ad_Name__Facebook_Ads'].tolist()
    ad_names = list(set(ad_names))

    # List of image paths
    image_paths = []

     # Iterate through each ad name and find corresponding images
    for ad_name in ad_names:
          image_name = f'{ad_name}.jpg'
          image_paths.append(image_name)
    return image_paths
    

# Function to create columns and display images with captions
def display_images(images, captions):
    num_images = len(images)
    cols = st.columns(num_images + 2)  # Extra columns for white space

    # Display images in the center columns
    for idx, image_name in enumerate(images):
        # Download the image from GCS to a temporary file
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
  past_test_data['Ad_Set'] = past_test_data['Ad_Set'].apply(lambda x: x.strip("'"))
  past_test_data = past_test_data.iloc[::-1].reset_index(drop=True)
  
  # Renaming columns in a DataFrame
  data = data.rename(columns={
      'Campaign_Name__Facebook_Ads': 'Campaign',
      'Ad_Set_Name__Facebook_Ads': 'Ad_Set',
      'Ad_Name__Facebook_Ads' : 'Ad_Name',
      'Impressions__Facebook_Ads' : 'Impressions',
      'Link_Clicks__Facebook_Ads' : 'Clicks',
      'Amount_Spent__Facebook_Ads' : 'Cost',
      'Lead_Submit_SunPower__Facebook_Ads' : 'Leads',
      'Ad_Effective_Status__Facebook_Ads' : 'Ad_Status',
      'Ad_Preview_Shareable_Link__Facebook_Ads' : 'Ad_Link'
  })


  # Streamlit interface for selecting new ad set
  with st.expander('Update Current Test and Upload Images'):
      new_ad_set_name = st.text_input("Enter New Ad Set Name")
      uploaded_images = {}
      campaign_name = None

      if new_ad_set_name:
          # Retrieve ad names for the new ad set
          ad_names = get_ad_names(new_ad_set_name, st.session_state.full_data)

          if len(ad_names) == 0:
              st.error("There is no ad_set with this name")
          else:
              if len(ad_names) > 6:
                  campaign_name = st.text_input("Enter Campaign Name")
                  if campaign_name:
                      ad_names = filter_ad_names_by_campaign(new_ad_set_name, campaign_name, st.session_state.full_data)
                      if len(ad_names) == 0:
                          st.error("There is no campaign with this name")
                          return
                  else:
                      st.error("Please enter a campaign name to proceed.")
                      return
              # Display file uploaders for each ad name
              all_images_uploaded = True
              for ad_name in ad_names:
                  uploaded_file = st.file_uploader(f"Upload image for {ad_name}", key=ad_name, type=['png', 'jpg', 'jpeg'])
                  uploaded_images[ad_name] = uploaded_file

                  if uploaded_file is None:
                      all_images_uploaded = False

              if all_images_uploaded and st.button("Update Ad Set and Upload Images"):
                  update_ad_set_if_exists(new_ad_set_name, uploaded_images, st.session_state.full_data, bucket_name, campaign_name)

  if current_test_data.empty:
            st.markdown("<h4 style='text-align: center;'>No Current Tests to Display</h4>", unsafe_allow_html=True)
  else:              
            current_Ad_Set = current_test_data['Ad_Set'].iloc[0]
          
            current_Ad_Set = current_Ad_Set.strip("'")
          
            campaign_value = get_campaign_value(current_Ad_Set, current_test_data)
          
            if campaign_value:
                  # Filter data on both ad_set and campaign_value
                  ad_set_data = data[(data['Ad_Set'] == current_Ad_Set) & (data['Campaign'] == campaign_value)]
            else:
                  # Filter data on just ad_set
                  ad_set_data = data[data['Ad_Set'] == current_Ad_Set]
                    
            data = ad_set_data
                    
            selected_columns = ['Ad_Set', 'Ad_Name', 'Impressions', 'Clicks','Cost', 'Leads']
            filtered_data = data[selected_columns]
          
            # Grouping the data by 'Ad_Set'
            grouped_data = filtered_data.groupby(['Ad_Set', 'Ad_Name'])
            
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
          
            column_order = ['Ad_Set', 'Ad_Name', 'Cost', 'CPM', 'Clicks', 'CPC', 'CTR', 'Leads', 'CPL', 'CVR', 'Significance']
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
            #ct_images = final_df["Ad_Name"]
            #filtered_list = [item for item in ct_images if item is not None]
            
            final_adnames = final_df['Ad_Name']
            final_adnames = [item + ".jpg" for item in final_adnames]
            final_adnames.pop()
            
            display_images(final_adnames, final_adnames)        
          
  st.markdown("<h2 style='text-align: center;'>Past Tests</h2>", unsafe_allow_html=True)

  if past_test_data.empty:
            st.markdown("<h4 style='text-align: center;'>No Past Tests to Display</h4>", unsafe_allow_html=True)
  else:        
            past_tests = past_test_data['Ad_Set']
          
            # Dictionary to store DataFrames for each ad set
            ad_set_dfs = {}
            
            for ad_set in past_tests:
                ad_set_dfs[ad_set] = process_ad_set_data(st.session_state.full_data, ad_set, past_test_data)
          
            for ad_set in ad_set_dfs:
                with st.expander(f"Show Data for {ad_set}"):
                    st.dataframe(ad_set_dfs[ad_set], width=2000)
                    current_df = ad_set_dfs[ad_set]
                    ad_names = current_df['Ad_Name']
                    ad_names = [item + ".jpg" for item in ad_names]
                    ad_names.pop()
                    display_images(ad_names, ad_names)

if __name__ == '__main__':
    password_protection()
