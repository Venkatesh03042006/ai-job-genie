# Phase 4 Explainability Demo

Scanned `8` resumes to find `8` with a genuine (non-duplicate) chunk match against their top-ranked JD; `0` had none.

Split: `test` | matcher: `tfidf` | top_k: `3` | "match" = the matcher's own top-1 ranked JD from the full 2174-JD target-category pool (see module docstring) | exact-duplicate chunk pairs excluded


## Resume `resume-101` (Senior Software Engineer) <-> JD `job-3904095574` (Senior Software Engineer)

- **[0.294]** Your experience with 'Software Engineering Analyst, Software Engineering Analyst, Associate Software Engineer' matched the requirement 'Senior Software Engineer'
- **[0.044]** Your experience with 'C, SQL, Python, R, Tableau, HP ALM Quality center, HP QTP, MS office, Trello, Streak CRM, automation tools, business processes, Chemistry, CRM, client, English, event management, functional, HP, image, team lead, machinery, Mathematics, Oil, developer, Physics, Programming, progress, Project management, Quick Test Professional, Quality, quality assurance, simulation, VB script' matched the requirement 'Education and Length of ExperienceLevels 3, 45 years of related experience with a Bachelor’s degree in Computer Science, Electrical Engineering, Computer Engineering, Aerospace Engineering, Physics, Mechanical Engineering, Math, or related field3 years of related experience with a Masters’ degree in Computer Science, Electrical Engineering, Computer Engineering, Aerospace Engineering, Physics, Mec'

## Resume `resume-234` (Data Science Engineer) <-> JD `job-3901949696` (Data Science Engineer)

- **[0.531]** Your experience with 'SDE and Business Analyst Trainee' matched the requirement 'Business Analyst with Data Science - Only US Citizens, Green Card - Locals'
- **[0.307]** Your experience with 'Business Analyst, Data Analytics, Data Cleansing, Business Analysis, Risk Analysis, Statistical Analysis, Deep Learning, Machine Learning, Python, Numpy' matched the requirement 'Business Analyst with Data Science - Only US Citizens, Green Card - Locals'
- **[0.000]** Your experience with 'Looking for roles related to application development in Machine Learning.' matched the requirement 'Business Analyst with Data Science - Only US Citizens, Green Card - Locals'

## Resume `resume-3` (Business Development Executive) <-> JD `job-3902371214` (Data Science Engineer) _[cross-category match]_

- **[0.194]** Your experience with 'accounts payables, accounts receivables, Accounts Payable, Accounts Receivable, administrative functions, trial balance, banking, budget, bi, closing, Computer Applications, Credit, clients, Customer Service, data entry, delivery, driving, email, insurance, inventory, ledger, Access, Excel, Outlook, PowerPoint, Word, mortgage loan, Enterprise, policies, QuickBooks, Sales, sales reports, telecommun' matched the requirement 'Implement models for loan loss allowance, CECL, stress testing, new volume origination, line of credit utilization, and prepayment models for all products, including credit card, personal loan, student loan, auto loan, mortgage, and commercial loan.Maintaining documentation for key processes and model components across the team with a focus on standardization of processes that satisfy model risk m'
- **[0.174]** Your experience with 'Accountant, Accounts Receivable Clerk, Mortgage Underwriter, Commercial Auto Underwriter, Personal Auto Underwriter, Claims Examiner' matched the requirement 'one (1) of the following areas is preferred; real estate products, auto, credit card, student loan, or commercial loan.'
- **[0.000]** Your experience with 'To obtain a position in a fast-paced business office environment, demanding a strong organizational, technical, and interpersonal position utilizing my skills and attributes.' matched the requirement 'Jr. Data Scientist (Hybrid)'

## Resume `resume-254` (Data Engineer) <-> JD `job-3901394057` (Data Science Engineer) _[cross-category match]_

- **[0.702]** Your experience with 'Deputy Data Analyst' matched the requirement 'Data Scientist/Data Analyst'
- **[0.489]** Your experience with 'Data Analyst, interested in Machine Learning application development. As a data analyst, I have extensive experience with data preprocessing and pipeline building and I would like to venture further into this domain to learn more about it.' matched the requirement 'Data Scientist/Data Analyst'
- **[0.219]** Your experience with 'Data Analyst, Data Analytics, Data Analysis, Time Series Analysis, Business Analytics, Predictive Modeling, Regression Analysis, Image Processing, Data Visualization, Linear Regression, Statistical Analysis, Requirement Analysis, SQL, Tableau, Python' matched the requirement 'Data Scientist/Data Analyst'

## Resume `resume-148` (Senior Software Engineer) <-> JD `job-3901933164` (Business Development Executive) _[cross-category match]_

- **[0.202]** Your experience with 'accounting, balance sheet, budgets, client, clients, derivatives, drafting, equity, financial, financial accounting, financial statements, fixed assets, Funds, Government, Information Technology, inventory, investments, ledger, MA, Microsoft Excel, natural, page, payables, processes, programming, Real Estate, research, sales, scheme, telephone, writing skills' matched the requirement 'ResponsibilitiesLead Generation: Inject your boundless energy into creating and managing events with our existing clients, connecting with financial advisors and realtors to ignite business growth.Networking: Be the life of the party as you build and nurture relationships with key partners in the financial and real estate industries.Classes and Seminars: Bring the house down (figuratively!) by setting up and organizing classes to educate the public on real estate investment, all while representing our expertise.QualificationsHigh Energy: An abundance of enthusiasm, dynamism, and a go-getter attitude.Excellent Communication: Strong verbal and written communication skills.Passion for Real Estate and Finance: A genuine interest and enthusiasm for the real estate and financial industries.Independence: Ability to work autonomously, setting and achieving goals.Networking Skills: Effective at building and maintaining professional relationships.Event Management: Experience or willingness to organize and manage events.Social Media expert.Compensation$28-$36 an Hour depending on experience.'
- **[0.000]** Your experience with 'Senior Accountant, Associate Fund Controller, Advisory, Forensic and Audit Associate, Accounting Tutor' matched the requirement 'Business Development Manager'

## Resume `resume-126` (Data Science Engineer) <-> JD `job-3901945212` (Data Engineer) _[cross-category match]_

- **[0.256]** Your experience with 'Machine Learning, Artificial Intelligence, Python, Predictive Analytics, Statistical Modeling, Data Visualization, Data Analysis, Data Mining, Data Validation, Power Bi, Text Analytics, Data Modeling, Data Analyst' matched the requirement ' multi-tenant, OLTP data modeling, dimensional data modeling, composite modeling, data transformation, row-level security, and designing the most optimal analytical data structures for near real-time data analyticsAdditional programming experience is a plus (preferably.NET) or other languages such as Python, Scala, R.'
- **[0.108]** Your experience with 'As a Data Analyst I always look into more innovative ways of finding figures and facts from data. I have also worked with some production based Machine learning application problems and I would like to involve myself more with such ventures.' matched the requirement 'You will be involved in the design and development efforts for our big data solutions including data lake, Business Intelligence Solutions, Machine Learning, Data Pipeline and cloud-based data warehouse products.'

## Resume `resume-197` (Data Science Engineer) <-> JD `job-3889751839` (Data Science Engineer)

- **[0.172]** Your experience with 'A python programmer, currently pursuing Data science and looking for roles that involve the application of Machine Learning. I am always excited to take on real life problems and tread through different technology stacks.' matched the requirement 'Architect cutting-edge Machine Learning solutions.Champion commercial driven Data Science.Nurture a culture of innovation within your team.Mentor emerging talent within the Data Science organization.Produce scalable machine learning algorithms.'
- **[0.162]** Your experience with 'Junior Developer' matched the requirement 'Leadership & Mentorship: Lead and nurture junior data analysts and datascientists, fostering their professional growth and integration into thevibrant culture of our Data Science organization.'
- **[0.142]** Your experience with 'Python Developer, Django, Flask, Data Science, Spark, PySpark, Machine Learning, Data Modelling, Natural language Processing, SVM, Computer Vision, Neural Networks.' matched the requirement 'Architect cutting-edge Machine Learning solutions.Champion commercial driven Data Science.Nurture a culture of innovation within your team.Mentor emerging talent within the Data Science organization.Produce scalable machine learning algorithms.'

## Resume `resume-12` (Data Science Engineer) <-> JD `job-3902371214` (Data Science Engineer)

- **[0.396]** Your experience with 'Data Analytics, Linear Regression, Logistic Regression, Business Intelligence, Business Analysis, GraphQL, Python' matched the requirement '3+ years of experience in quantitative modeling, development, or implementation.Working experience in data manipulation and advanced data analysis.Experience with SAS, R, Python, and proficiency working with large datasets is required.Applied experience with Logistic Regression, Linear Regression, Survival Analysis, Time Series Analysis, Decision Trees, and Cluster Analysis.Experience in at least '
- **[0.000]** Your experience with 'Associate Consultant, Junior Analyst Intern' matched the requirement 'Jr. Data Scientist (Hybrid)'
